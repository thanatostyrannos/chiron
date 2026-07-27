---
title: Tensors and autograd — layout, the reverse-mode graph, and the first place memory bites
version: 1.0.0
date: 2026-07-26
track: A — Foundations
prereqs: none (assumes Python fluency and the lab venv from ENVIRONMENT.md)
difficulty: 2/5 conceptually, 3/5 in the details
time: ~90 min reading, ~3 h exercises; splittable across three evenings
---

# Tensors and autograd

## What this module settles

A tensor is not an array; it is a **view specification over a flat buffer** — dtype,
shape, strides, offset — and almost every "reshape" in a model is address arithmetic
that moves zero bytes, except for the handful that move all of them, and you need to
know which is which. Reverse-mode autodiff is a **recorded dependency graph plus a
reverse traversal that accumulates into per-parameter buffers**, and the reason it
retains tensors from the forward pass is a two-line consequence of the matmul
derivative, not an implementation choice. Those retained tensors — the activations —
are the first place in this curriculum where memory becomes the binding constraint,
and at nanoGPT scale they are already **an order of magnitude larger than the model,
gradients, and optimizer state combined** — computed at 10–12×, measured on this machine
at 4.5–18× depending on batch size and dtype `[M]` (§2.5 and Exercise 2).

---

## 1. Theory in plain language

### What a tensor actually is

You have spent thirty years with the distinction between *a block device* and *a
filesystem's view of it*. PyTorch has the same split and it is the single most
useful thing to internalise on day one.

- **`torch.Storage`** — a flat, one-dimensional, untyped-ish allocation. A slab.
- **`torch.Tensor`** — a *descriptor* over that slab: `(storage, dtype, offset, shape,
  strides)`. Multiple tensors can describe the same slab, overlapping, at different
  offsets, with different shapes.

So `x.transpose(0,1)` is not a data movement. It is editing two small tuples. Same
storage, same `data_ptr()`, different metadata. This is `mmap` of the same file at two
different views, or a logical volume presenting a different geometry over the same
extents. It is metadata-only, and it is O(1) regardless of tensor size.

The break in that analogy comes fast. A filesystem can present *any* view over extents
because it has an indirection table per block. A PyTorch tensor has exactly `ndim`
integers of indirection — one stride per dimension — and nothing else. So the set of
views expressible without copying is small and rigid. When you ask for a view outside
that set, PyTorch does not silently build a mapping table; it raises, and you must call
`.contiguous()`, which is a **full read-and-write of the tensor at memory bandwidth**.
That is the layout tax, it is invisible in the source, and on our machine we know
exactly what it costs because we measured the bandwidth `[M]`.

### What replaced what

Before autodiff frameworks, you wrote the backward pass by hand. That is not a quaint
historical note — it is why architecture research was slow. Every new layer needed a
derivation, an implementation, and a finite-difference gradient check, and a sign error
in the backward pass produces a model that trains *almost* correctly, which is the worst
possible failure mode.

Reverse-mode automatic differentiation `[C]` (`1502.05767`, Baydin et al., 20 Feb 2015 —
the standard survey) removes the derivation by composing derivatives of *primitives*.
You register the derivative of `matmul`, `softmax`, `exp` once; every model built from
them differentiates for free. Two generations of framework:

- **Static graph** (TensorFlow 1.x, Theano): declare the graph, then run it. Optimizable,
  but debugging is a build-time/run-time split and control flow is painful.
- **Define-by-run / tape** (PyTorch, JAX's tracing): the graph is recorded *as the
  forward pass executes*. `if` statements are just `if` statements. This won, and it is
  the only model you need to hold.

The cost of define-by-run is exactly the thing this module is about: to run the reverse
pass you must **keep the forward pass's intermediate tensors alive**. That is the
activation-memory bill, and nobody sends it to you itemised. Exercise 2 builds the
itemiser.

### Systems bridge, and where it breaks

| Systems concept | Autograd counterpart | Where it breaks |
|---|---|---|
| Write-ahead log: append records, replay in order | The graph records each op; backward replays in reverse topological order | The graph is **not durable and not replayable twice**. `backward()` frees the saved tensors as it consumes them; a second call fails. It is a one-shot tape, not a log. |
| Read-modify-write accumulator | `p.grad += ...` on every contributing edge | There is no compare-and-swap and no idempotence. Running backward twice without `zero_grad` silently doubles the gradient — a correctness bug that produces plausible-looking loss curves. |
| Checkpoint interval in a WAL replay window | Activation checkpointing: store every k-th layer, recompute the rest | You choose a WAL interval from RPO/RTO and pay recovery cost only after a failure. Here you pay the recompute on **every single step**, forever. There is no failure-free path. |
| Cache with a miss handler | Recompute of a discarded activation | The recompute is **exact and always available** — unlike a KV cache at inference, where eviction is destruction (`research/memory/kv-cache-mechanics.md` §2). Training-time activations are the one case in this whole curriculum where the "backing store" genuinely exists. |

That last row is the load-bearing one, and it lines up with the memory track's central
finding: `research/memory/README.md` concludes that **reconstructibility, not speed, is
the distinction that partitions the things called "memory."** Activations are perfectly
reconstructible. KV entries are reconstructible only by re-running prefill over the whole
prefix. Session stores are not reconstructible at all. Same word, three different
guarantees.

---

## 2. The math that actually matters

### 2.1 The layout equation

A tensor's element at logical index `(i₀, i₁, …, i_{n-1})` lives at

```
element_index = offset + Σ_k  i_k · s_k
byte_address  = storage_base + element_index · itemsize
```

| Symbol | In words |
|---|---|
| `i_k` | the index along dimension `k` — the k-th subscript you wrote |
| `s_k` | the **stride** of dimension `k`: how many *elements* (not bytes) you skip in the flat buffer to advance that subscript by one |
| `offset` | where this tensor's element zero sits inside the shared storage; nonzero for slices |
| `itemsize` | bytes per element (fp32 → 4, bf16 → 2, bool → 1) |

**Worked.** A contiguous float32 tensor of shape `(2, 3, 4)`. Row-major (C order) strides
are the suffix products of the shape: `s₂ = 1`, `s₁ = 4`, `s₀ = 3·4 = 12`. So
`stride = (12, 4, 1)`. Element `[1, 2, 3]` sits at `1·12 + 2·4 + 3·1 = 23`, i.e. byte
`92`. There are 24 elements, so index 23 is the last one — which is correct, `[1,2,3]` is
the final element.

Now `t.transpose(0, 1)`. Shape becomes `(3, 2, 4)`; the stride tuple is permuted the same
way, to `(4, 12, 1)`. No bytes moved. Verify: the element that was `[1,2,3]` is now
`[2,1,3]`, at `2·4 + 1·12 + 3·1 = 23`. Same address. ✓

**Contiguity** means the strides are exactly the suffix products of the shape — the
logical iteration order and the physical order agree. After the transpose they do not:
`s₀ = 4 < s₁ = 12`. Walking the transposed tensor in logical order visits flat offsets

```
0, 1, 2, 3,  12, 13, 14, 15,  4, 5, 6, 7,  16, 17, 18, 19,  8, 9, 10, 11,  20, 21, 22, 23
```

which is not an arithmetic progression, so **no single stride can describe a flat
24-element view of it**. `.view(24)` must fail; `.reshape(24)` succeeds by silently
copying. That silence is the trap.

**Worked on real code.** `research/reference/training/nanogpt/model.py:57`:

```
x            : (B, T, C)            = (64, 256, 384),  stride (98304, 384, 1)
.view(B,T,nh,hs)                    = (64, 256, 6, 64), stride (98304, 384, 64, 1)   contiguous
.transpose(1,2)                     = (64, 6, 256, 64), stride (98304, 64, 384, 1)   NOT contiguous
```

That transpose is how "split into heads" is implemented everywhere: it is free. Then
attention runs, and `model.py:72` has to put the heads back side by side:

```
y            : (64, 6, 256, 64), stride (98304, 16384, 64, 1)   contiguous
.transpose(1,2)                  = (64, 256, 6, 64), stride (98304, 64, 16384, 1)
.view(B,T,C) would need            (64, 256, 384),   stride (98304, 384, 1)   — impossible
```

hence the `.contiguous()` sitting in the middle of that line. **Cost of that one call,**
at the shakespeare_char config (`B=64, T=256, C=384`, bf16):

```
elements        = 64 × 256 × 384          = 6,291,456
bytes           = 6,291,456 × 2           = 12,582,912  B = 12.0 MiB
traffic (r + w) = 2 × 12,582,912          = 25,165,824  B = 24.0 MiB   per layer
6 layers        = 150,994,944 B = 144 MiB per forward pass
```

At our measured `[M]` 199.9 GB/s device-to-device copy bandwidth
(`notebook/uma-carveout-controls-fast-tier.md`, single run):
`150,994,944 / 199.9e9 = 0.755 ms` per forward, and the backward pass performs the
mirror-image copy. That is a **floor**, not a prediction: a strided gather will not reach
peak streaming bandwidth. Exercise 1 measures the real number and the gap is the finding.

### 2.2 Broadcasting

Rule, stated once. Right-align the two shapes. For each aligned pair of dimensions,
either they are equal, or one of them is 1, or it is an error. The output dimension is
the max. Missing leading dimensions are treated as 1.

The mechanism is the part worth knowing: **a broadcast dimension is given stride 0.** The
index arithmetic `Σ i_k · s_k` then contributes nothing for that subscript, so every value
of `i_k` reads the same physical element. Broadcasting is repetition by address
arithmetic. Zero bytes copied, zero allocation.

**Worked.** `model.py:179`, `tok_emb + pos_emb`:

```
tok_emb : (b, t, n_embd)                            e.g. (64, 256, 384)
pos_emb : (t, n_embd)  → treated as (1, t, n_embd)  stride[0] set to 0
result  : (64, 256, 384)                            newly allocated, 12 MiB in bf16
```

The broadcast itself is free. The *result* is not — every elementwise op materialises a
full output tensor. Broadcasting saves you the input copy and never the output.

**Second worked case, and the one that matters for memory.** `model.py:68`:

```
self.bias[:,:,:T,:T]  : (1, 1, 256, 256) bool     = 64 KiB
att                   : (B, nh, T, T)             = (64, 6, 256, 256)
                      = 25,165,824 elements       = 48.0 MiB in bf16, per layer
```

A 64 KiB mask broadcasts against a 48 MiB score matrix. The mask is trivial; the thing it
is masking is the quadratic activation term that dominates §2.5.

**The adjoint of a broadcast is a sum.** If `y = x.expand(m, n)` from `x` of shape
`(1, n)`, then for the loss `L`:

```
x̄[0, j] = Σ_{i=0}^{m-1}  ȳ[i, j]
```

In words: because `x[0,j]` was read by `m` different output positions, its gradient is
the sum of the gradients arriving from all `m` of them. So the free forward broadcast
becomes a real `m·n`-element reduction in the backward pass. The cost was deferred, not
removed.

**Where the analogy breaks.** A stride-0 view looks like a copy-on-write page mapping, and
on the read path it behaves like one. But there is no fault-on-write. An in-place write
through a broadcast view has many logical indices aliasing one physical element; the
result is order-dependent garbage, not a COW split. PyTorch defends this partly (`expand`
returns a tensor you are warned not to write to) and partly not at all (`+=` on the result
of a broadcasting op is a routine footgun). The mapping is one-way in a way virtual memory
never is.

### 2.3 Reverse-mode autodiff, from the chain rule

Let the computation be a directed acyclic graph of primitive ops ending in a **scalar**
loss `L`. For any intermediate tensor `v`, define its **adjoint**:

```
v̄  ≡  ∂L / ∂v          (same shape as v)
```

In words: `v̄[i]` is the number such that nudging `v[i]` up by a tiny `ε` changes the loss
by approximately `v̄[i]·ε`. It is a sensitivity, one per element, and it has exactly the
same shape as the thing it is about. Hold that: **every gradient has the shape of its
tensor.** That fact alone resolves most shape confusion in backward passes.

The seed is `L̄ = ∂L/∂L = 1`. That is what the bare `loss.backward()` means — it is
`loss.backward(torch.tensor(1.0))` with the default filled in, and it is why `backward()`
on a non-scalar demands an explicit argument: there is no canonical seed.

For a node `y = f(x₁, …, x_k)`, each input receives

```
x̄ᵢ  +=  Jᵢᵀ · ȳ        where  Jᵢ = ∂y/∂xᵢ
```

| Symbol | In words |
|---|---|
| `Jᵢ` | the **Jacobian** of this op with respect to input `i`: one row per element of the output, one column per element of the input |
| `Jᵢᵀ · ȳ` | the **vector-Jacobian product** (VJP): "push the output sensitivities backwards through this op" |
| `+=` | accumulate, do not assign — §2.4 explains why this is a `+=` and not an `=` |

Reverse mode **never materialises `Jᵢ`.** For a linear layer with 4096×4096 weights the
Jacobian would have 2.8×10¹⁴ entries. Each primitive instead implements the VJP directly
as a couple of tensor ops. `Jᵀ·ȳ` is notation for a routine, not a matrix multiply against
a stored matrix.

**Why reverse and not forward.** Forward mode propagates one *input* perturbation through
the whole graph, giving you one column of the Jacobian per pass. Reverse mode propagates
one *output* sensitivity backwards, giving you one row per pass. Training has ~10⁸ inputs
(parameters) and exactly 1 output (the loss). Forward mode would need ~10⁸ passes; reverse
mode needs one. That ratio — inputs to outputs — is the whole argument, and it is why the
field standardised on reverse mode despite its memory cost `[C]` (`1502.05767`).

### 2.4 The two VJPs you must know by heart

**Matmul.** `Y = X W` with `X : (m, k)`, `W : (k, n)`, `Y : (m, n)`.

```
X̄ = Ȳ Wᵀ        shape (m,n)·(n,k) → (m,k)   ✓ same shape as X
W̄ = Xᵀ Ȳ        shape (k,m)·(m,n) → (k,n)   ✓ same shape as W
```

Count the arithmetic. A GEMM of `(a,b)·(b,c)` costs `2·a·b·c` FLOPs (one multiply and one
add per inner-product term).

```
forward   :  2·m·k·n
backward  :  2·m·n·k  +  2·k·m·n  =  4·m·k·n
total     :  6·m·k·n  =  3 × forward
```

**Backward is exactly 2× forward for the dense linear algebra.** This is where the famous
`6·N·D` training-FLOPs rule comes from: `2N` per token forward, `4N` backward, `6N` total,
times `D` tokens. Inference is `2N` per token. Training a model costs 3× what running it
once costs, per token. That number carries into the scaling-laws module unchanged.

Now the memory consequence, and it is two lines and it explains everything downstream:

> `W̄ = Xᵀ Ȳ` requires **X**, the layer's input.
> `X̄ = Ȳ Wᵀ` requires **W**, which is a parameter and resident anyway.

So for every matmul in the network, the *input activation* must survive from the forward
pass until the backward pass reaches that node. Parameter memory is `O(params)`. Activation
memory is `O(params_touched × batch × sequence)` — it scales with the data, not the model.
That asymmetry is not a design decision anyone made. It falls out of the derivative of a
product.

**Softmax.** With `p = softmax(z)` along the last axis, the Jacobian is
`∂pᵢ/∂z_j = pᵢ(δᵢⱼ − p_j)` where `δᵢⱼ` is 1 when `i = j` and 0 otherwise. The VJP collapses
to

```
z̄ⱼ = pⱼ · ( p̄ⱼ − Σᵢ p̄ᵢ pᵢ )
```

In words: the gradient flowing into score `j` is its own probability, times the difference
between the gradient arriving at that probability and the probability-weighted average of
all gradients arriving in that row. Note what it needs: **`p`, the output — not `z`, the
input.** Softmax saves its output for backward. Every nonlinearity saves either its input
or its output; none of them saves nothing.

This is exactly the hook FlashAttention exploits. If you save the per-row log-sum-exp
(one fp32 scalar per query position) instead of the full `(B, nh, T, T)` probability
matrix, you can regenerate `p` tile-by-tile inside the backward kernel. The memory ratio,
bf16 scores versus fp32 statistics:

```
B·nh·T²·2  /  B·nh·T·4  =  T / 2
```

At `T = 256` that is a **128× reduction** in attention-specific saved state; at `T = 32768`
it is 16,384×. `research/reference/training/nanogpt/model.py:62` is the branch that
chooses between these two worlds, and Exercise 2 measures both sides of it.

> `[M]` **On our machine, that branch does nothing by default — and this was measured
> while writing this module.** Run 2026-07-26, gfx1151, torch
> `2.12.0a0+rocm7.13.0a20260313`, single run, `B=16`, `L=6`, `nh=6`, fp32, fitting
> `saved(T) = k + a·T + c·T²` over `T ∈ {64, 128, 256}`:
>
> | Path | quadratic coefficient `c` | activation bytes at `T=256` |
> |---|---|---|
> | manual (`model.py:67–71`) | 2,310 B/token² | 733.8 MiB |
> | SDPA (`model.py:64`), default env | 2,304 B/token² | 733.5 MiB |
> | SDPA, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | **0.0 B/token²** | **590.0 MiB** |
>
> The predicted coefficient for exactly one saved `(B, nh, T, T)` fp32 tensor per layer is
> `6 × 16 × 6 × 4 = 2,304`, so the default SDPA path is materialising the score matrix
> **identically to the manual path**. PyTorch reports `flash_sdp_enabled() == True` and
> `mem_efficient_sdp_enabled() == True`, but emits a runtime warning that both AMD
> backends are experimental and gated behind that environment variable, and silently
> dispatches to the math backend. Only with the variable set does the `(16, 6, 256, 256)`
> tensor leave the saved-tensor ledger, replaced by `(16, 6, 256, 64)` head-shaped
> tensors. The mechanism checks out arithmetically: the saving is `733.5 − 590.0 =
> 143.5 MiB`, against a predicted score-tensor size of `6 layers × 16 × 6 × 256² × 4 B =
> 144.0 MiB`. Single run; an anecdote by the house standard, but the coefficient is exact
> to four significant figures and the effect is binary. **This belongs in
> `ASSUMPTIONS.md` as a new row.**

### 2.5 Activation memory, in numbers

The standard published accounting `[C]` (`2205.05198`, Korthikanti et al., 10 May 2022 —
"Reducing Activation Recomputation in Large Transformer Models") gives, per transformer
layer, with 16-bit storage, no tensor/sequence parallelism and no recomputation:

```
bytes_per_layer = s · b · h · ( 34 + 5·a·s / h )
```

| Symbol | In words |
|---|---|
| `s` | sequence length (tokens per sample) |
| `b` | micro-batch size (samples) |
| `h` | hidden size / model width |
| `a` | number of attention heads |
| `34` | the count of bytes retained per element of the `(s,b,h)` residual stream — roughly 17 two-byte tensors per layer (QKV inputs, MLP intermediates, norm outputs, dropout masks) |
| `5·a·s/h` | the quadratic term: **5 bytes per element of the `(b,a,s,s)` attention matrix** — two bf16 tensors (softmax input, softmax output) plus a one-byte dropout mask |

Sanity-check that decomposition yourself: `s·b·h · 5·a·s/h = 5·a·b·s²`, which is exactly
5 bytes per attention-matrix element. The formula is not magic; it is a tensor inventory.

> `[M]` **And the inventory is out of date, which Exercise 2 shows directly.** Measured on
> this stack with dropout off, current PyTorch autograd saves **one** tensor of attention-
> matrix shape per layer, not two: `2,304 = L·b·a·4` bytes per `s²` in fp32, i.e. 4 bytes
> per element, i.e. 2 bytes in bf16. The softmax VJP (§2.4) needs only the softmax
> *output*, so the pre-softmax scores are never saved; `masked_fill` backward needs only
> the mask. Korthikanti's `5 = 2 + 2 + 1` therefore over-counts the quadratic term by
> roughly 2× for a modern implementation with dropout off. Turn dropout back on and the
> `+1` byte mask reappears, giving 3 rather than 5. **Do not use the published constant
> for capacity planning on this stack; use the profiler.** This is a small, concrete
> instance of the general problem in §7.

**Worked at nanoGPT shakespeare_char scale** (`config/train_shakespeare_char.py:22`:
`n_layer=6, n_head=6, n_embd=384`, `block_size=256`, `batch_size=64`):

```
s·b·h        = 256 × 64 × 384          = 6,291,456
5·a·s/h      = 5 × 6 × 256 / 384       = 20
per layer    = 6,291,456 × (34 + 20)   = 339,738,624 B = 324.0 MiB
6 layers                               = 2,038,431,744 B = 1.90 GiB
   of which quadratic  6,291,456 × 20 × 6 = 754,974,720 B = 720 MiB
```

Against the persistent training state. Parameter count, computed from the config rather
than quoted: embeddings `65·384 + 256·384 = 123,264`; per block
`2·768 (LayerNorms) + 443,520 (c_attn) + 147,840 (c_proj) + 591,360 (mlp fc) + 590,208
(mlp proj) = 1,774,464`; six blocks `= 10,646,784`; final norm `768`. Total
**10,770,816 parameters** (`get_num_params()` reports 10,672,512 because it subtracts
`wpe`, `model.py:159`).

```
params      fp32  10,770,816 × 4  =  43.1 MB  =  41.1 MiB
gradients   fp32                  =  41.1 MiB
AdamW m, v  fp32                  =  82.2 MiB
persistent total                  = 164.4 MiB
```

Now the comparison, stated at the precision each number deserves:

| Activation figure | Source | vs 164.4 MiB persistent |
|---|---|---|
| 1.90 GiB, `b=64`, bf16, dropout on | `[C]` Korthikanti formula above | 11.8 × |
| ~1.62 GiB, same, corrected to 3 bytes/attention-element | `[C]` formula + `[M]` our inventory correction | 10.1 × |
| **733.8 MiB**, `b=16`, `T=256`, **fp32**, dropout off | `[M]` Exercise 2, this machine, single run | **4.5 ×** |
| 2.87 GiB, the same measurement scaled linearly to `b=64` | `[M]` × 4 (activations are linear in batch) | **17.9 ×** |

**At the smallest published transformer recipe in the reference library, activations are
an order of magnitude larger than the model, gradients, and optimizer state combined.**
That is the sentence to carry out of the module; the exact multiple depends on batch size,
dtype, dropout, and which attention kernel you got, and the spread above — 4.5× to 17.9×
across reasonable choices — is itself the lesson. It is also why activation memory, not
parameter memory, sets your maximum micro-batch, and therefore your throughput, and
therefore whether an ablation arm is affordable.

Three honest caveats, all important:

1. The formula predates fused attention. With `scaled_dot_product_attention`
   (`model.py:64`) the 720 MiB quadratic term largely disappears and is replaced by an
   `O(s)` log-sum-exp term. Nobody has published a maintained successor formula covering
   fused kernels, `torch.compile` fusion, and selective recompute — see §8.
2. It assumes dropout is on (the `+1` byte mask). `train_shakespeare_char.py:25` sets
   `dropout = 0.2`, so it is — but Exercise 2 turns it off to get a clean ledger, which is
   part of why the measured and computed rows in the table above are not comparable
   line-for-line.
3. `[M]` Its quadratic term over-counts by ~2× against what current PyTorch autograd
   actually saves (the note above). Both the formula and the profiler are in this module
   so that you can see the size of the gap between a published constant and your own
   machine, which is generally larger than people assume.

**The store-versus-recompute trade, with the arithmetic.** `[C]` (`1604.06174`, Chen et al.,
21 Apr 2016 — "Training Deep Nets with Sublinear Memory Cost"). Take `n` layers, keep only
every `k`-th layer's input, and recompute the rest during backward. Peak activation
storage is `n/k` checkpoints plus the `k` layers of a single segment being recomputed:

```
f(k) = n/k + k        f'(k) = −n/k² + 1 = 0        k = √n        f(√n) = 2√n
```

So memory drops from `O(n)` to `O(√n)` at the cost of exactly one extra forward pass. In
FLOPs: baseline is `1 forward + 2 backward = 3` units, plus one recomputed forward makes
`4` units — **+33% compute for a `n / (2√n) = √n/2` memory reduction.**

- `n = 48` (Laguna-S depth): `k = 7`, peak `≈ 48/7 + 7 = 13.9` units versus 48 → **3.5×
  less**, for +33% time. Clearly worth it.
- `n = 6` (nanoGPT): `k ≈ 2.4`, peak `≈ 5` units versus 6 → **1.2× less**, for +33% time.
  Clearly not worth it.

The technique only pays at depth, which is a useful thing to know before you turn it on
reflexively.

### 2.6 Why gradients accumulate

Two independent reasons, routinely conflated.

**Reason one — the graph says so.** If a tensor `v` feeds *more than one* consumer, the
multivariate chain rule requires summing the contributions:

```
v̄  =  Σ_{c ∈ consumers(v)}  (∂y_c/∂v)ᵀ · ȳ_c
```

In words: a tensor used in three places influences the loss through three routes, and its
total sensitivity is the sum over routes. This is not an API convenience — it is the
chain rule for a DAG rather than a chain. Two places it shows up in nanoGPT:

- **Residual connections** (`model.py:105`, `x = x + self.mlp(self.ln_2(x))`): `x` feeds
  both the MLP branch and the skip. Its adjoint is the sum. This is also, incidentally, the
  mechanical reason residual networks train — the skip route contributes an identity term
  to every adjoint, so gradient magnitude does not have to survive a product of many
  Jacobians.
- **Weight tying** (`model.py:138`, `self.transformer.wte.weight = self.lm_head.weight`):
  one parameter tensor, two use sites — the embedding lookup at the bottom and the output
  projection at the top. `.grad` receives both contributions. Nothing in the code says
  "add these together"; the graph does it because the same `Parameter` object has two
  edges into it.

**Reason two — `.grad` is a persistent buffer that `backward()` adds into.** This is an API
decision, and it is what makes gradient accumulation across micro-batches work
(`train.py:292–305`). It is a read-modify-write accumulator with no ownership protocol:
forget `optimizer.zero_grad()` (`train.py:314`) and you get the sum of two steps' gradients
with no error, no warning, and a loss curve that looks slightly off rather than broken.

**The normalisation subtlety, and it is a real difference between two production
codebases.** With `M` micro-batches you want the gradient of the *whole-batch mean* loss.
nanoGPT divides each micro-batch's mean loss by `M` (`train.py:301`):

```
g_nanogpt  =  (1/M) Σ_j  ∇ ( (1/nⱼ) Σ_{t ∈ batch j} ℓ_t )
```

where `nⱼ` is the token count of micro-batch `j` and `ℓ_t` is the per-token loss. That is a
mean of means. It equals the true mean `∇( (1/Σnⱼ) Σ_all ℓ_t )` **only when every `nⱼ` is
equal.** OLMo-core instead divides by the whole-batch token count *before* backward
(`loss_div_factor = batch_num_tokens_for_loss`, see the "Worth knowing" note in
`research/reference/CODE_MAP.md` for the OLMo-core section), which is exact for ragged
micro-batches. With variable-length sequences, document packing, or label masking, `nⱼ`
varies, and the mean-of-means silently up-weights short micro-batches. Exercise 3 measures
the error.

### 2.7 `detach`, `no_grad`, and the difference that matters

| Operation | What it does | Effect on memory |
|---|---|---|
| `x.detach()` | Returns a new tensor **sharing `x`'s storage**, with `requires_grad=False` and `grad_fn=None`. Cuts exactly one edge. | Frees nothing by itself. It frees the upstream graph only if nothing else still references it. |
| `with torch.no_grad():` | No graph nodes are recorded at all inside the block. No saved-tensor references are taken. | **This is the one that saves memory reliably.** Intermediates are freed as soon as they go out of scope. |
| `p.requires_grad_(False)` | This parameter is a leaf that will not receive a gradient. | Saves the `.grad` buffer (one full parameter-sized tensor), and prunes graph edges that exist only to reach it. |
| `optimizer.zero_grad(set_to_none=True)` | Sets `.grad = None` rather than writing zeros. | **Frees a model-sized allocation** and skips a full write pass. This is why `set_to_none` is the default now. |
| `loss.item()` | Reads a device scalar to the host. | Costs nothing in memory and everything in pipeline: a full device→host synchronisation. `train.py:321` says so in a comment. |

The canonical training-loop memory leak, and it is in production source as the *fix*:

```python
ce_batch_loss += get_local_tensor(ce_loss.detach())     # train_module.py:414
del ce_loss                                             # train_module.py:415
```

Without `.detach()`, `ce_batch_loss` holds a live reference to micro-batch `j`'s loss node,
which transitively pins that micro-batch's entire activation graph. Accumulate over `M`
micro-batches and peak activation memory is `M ×` what it should be — the exact thing
micro-batching exists to avoid. The bug is invisible until you OOM, and the fix is one
method call. Learn to see it.

`torch.no_grad()` is why `estimate_loss` (`train.py:215`) can run at far larger effective
batch than training does, and why `generate` (`model.py:305`) can run at all. It is not an
optimisation; without it, sampling 500 tokens would build a 500-step graph.

---

## 3. Why it matters for Proteus

**The config surface is an activation-memory surface.** Every axis we intend to ablate —
`n_layer`, `n_head`, `n_kv`, `head_dim`, `sliding_window`, MoE `top_k`, sequence length —
moves the activation bill, and the activation bill sets the maximum micro-batch. That is a
confound waiting to happen: two arms at matched parameter count and matched token budget
can still differ in achievable micro-batch, and if the loss normalisation is
mean-of-means (§2.6) the arms are then not even computing the same objective. **Proteus's
training loop must denominate the loss in tokens, not micro-batches, before any ablation
runs.** This is a rig requirement derived here, not a preference.

**The layout tax is a real, measurable line item on our machine.** Any attention variant
that changes head layout — GQA repeat-interleave, MLA's latent expand, a per-head gate —
pays `.contiguous()` traffic. We know our bandwidth `[M]` (199.9 GB/s at a 62 GiB
footprint, `notebook/uma-carveout-controls-fast-tier.md`), so a layout change has a
predicted cost in milliseconds before we write the kernel. Predicting then measuring is the
attribution discipline this lab is built around; here is the cheapest possible instance
of it.

**Mnemosyne inherits the view-versus-copy question directly.** A KV-cache interface that
hands out *views* into a preallocated pool costs stride arithmetic. One that hands out
*copies* costs bandwidth proportional to the cache size. Worse: any eviction policy that
"compacts" the cache is a `.contiguous()` in disguise, and any policy that gathers a subset
of tokens (`index_select` on the sequence axis) is a strided read at exactly the access
pattern that misses peak bandwidth. **The eviction policy interface must therefore expose
whether a policy needs a compacted cache**, because "H2O costs 3% more perplexity" and
"H2O costs a 62 GiB memcpy per step" are different findings and only one of them is in the
papers.

**The 32 GiB single-tensor fault constrains attention shapes directly.** `[M]`
`ASSUMPTIONS.md: large-tensor-fault-32gib` — a 31 GiB buffer copies cleanly at 199.9 GB/s;
a 32 GiB buffer hangs silently at 0 CPU. A materialised bf16 attention score tensor is
`B · nh · T² · 2` bytes:

```
T = 32,768, nh = 8, B = 1  :  8 × 32768² × 2  = 17,179,869,184 B = 16.0 GiB   (clean)
T = 32,768, nh = 8, B = 2  :                    34,359,738,368 B = 32.0 GiB   (the fault)
solve at B = 1, nh = 8     :  T = sqrt(32·2³⁰ / 16) = 46,341 tokens
```

So a materialised attention path on this machine hits a **silent hang** at batch 2, 32k
context. Not an OOM — a hang at zero CPU, which a long run would experience as a stall
rather than a crash.

**And by default we are on the materialised path.** The `[M]` measurement in §2.4 shows
that calling `scaled_dot_product_attention` on gfx1151 with this wheel dispatches to the
math backend and allocates the full `(B, nh, T, T)` tensor, exactly as the manual code
does; only `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` moves it to the memory-efficient
kernel. Three consequences, and they are load-bearing for the ablation plan:

1. **Any long-context arm must set that environment variable, and must state that it did.**
   Otherwise the fault surface `B · T² = 2³¹` (at `nh = 8`, bf16) is live, and the failure
   mode is a stall rather than an error. `scripts/activate-lab.ps1` is the right place for
   it, alongside the hipBLASLt settings — but it enables a backend AMD labels experimental,
   so it is also a numerics change, which means it must be set *before* the Hardware
   Validation Gate runs, not after, or the gate validates a configuration we do not use.
2. **Any activation-memory number measured without it is measuring the wrong machine**,
   and the error grows as `T²`. At `T = 256` it was 20% of total activations; at `T = 4096`
   it would be the overwhelming majority.
3. Any long-context experiment must allocate per layer rather than as one pooled tensor —
   the same conclusion `research/memory/open-problems-ranked.md` reaches for the KV pool.

`[A]` Medium confidence that the experimental backend is numerically sound enough to use;
CLAUDE.md already notes an undocumented AOTriton attention speedup reported for this
silicon and instructs us to verify whether it applies. The cheapest test is a bf16 and
fp32 attention-output comparison against the math backend at the shapes we run — which is
a Hardware Validation Gate numerics item that now has a specific reason to exist.

**Gradient correctness on gfx1151 is unproven, and this module is where we can cheaply
start proving it.** `ASSUMPTIONS.md: bf16-numerics-unproven` is `untested`. Exercise 3 runs
the same gradient computation on CPU fp32 and GPU fp32 and compares element-wise. That is
a Hardware Validation Gate contribution obtainable in under an hour, and it is strictly
cheaper than the full numerics suite.

---

## 4. Read the code

Paths are relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. Read in this order.

### Layout and broadcasting — `training/nanogpt/model.py`

| Where | What to look at, and why |
|---|---|
| `training/nanogpt/model.py:57` | `k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)`. The universal "split into heads" idiom, executed three times in three lines. Print `.stride()` before and after: `(98304, 384, 64, 1)` → `(98304, 64, 384, 1)`. Zero bytes move. Every attention implementation you will ever read does this. |
| `training/nanogpt/model.py:72` | `y.transpose(1, 2).contiguous().view(B, T, C)`. The one place in the attention block that actually copies, and the comment does not say so. This is the layout tax; §2.1 computes it at 144 MiB per forward for the shakespeare_char config. Ask yourself why `.reshape()` was not used — the answer is that `.reshape()` would have hidden the copy entirely. |
| `training/nanogpt/model.py:179` | `tok_emb + pos_emb` — a `(t, n_embd)` tensor broadcasting against `(b, t, n_embd)`. Stride-0 on the batch axis. Free on the input, allocates a full-size output. |
| `training/nanogpt/model.py:68` | `att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))` — a 64 KiB `(1,1,T,T)` mask broadcasting against a 48 MiB `(B,nh,T,T)` score tensor. Note that the comparison `== 0` materialises its own `(1,1,T,T)` bool. Small here; not small if you ever broadcast the mask eagerly. |
| `training/nanogpt/model.py:187` | `F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))`. Two `.view()` calls that succeed because `logits` is contiguous. Flattening `(B,T,V)` to `(B·T, V)` is free precisely because the strides allow it — the same operation on a transposed tensor would throw. |

### The autograd graph — `training/nanogpt/`

| Where | What to look at, and why |
|---|---|
| `training/nanogpt/model.py:138` | `self.transformer.wte.weight = self.lm_head.weight`. One `Parameter`, two consumers. Its `.grad` is the sum of the embedding-lookup contribution and the output-projection contribution. Multi-consumer accumulation (§2.6) made concrete in one line. |
| `training/nanogpt/model.py:267` | `param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}`. The optimizer is built only over leaves that will actually receive gradients. Read the following three lines too: the decay/no-decay split is by `p.dim() >= 2`, which is a shape-based heuristic standing in for "is this a matmul weight." |
| `training/nanogpt/model.py:305` | `@torch.no_grad()` on `generate`. Without it, sampling `max_new_tokens` tokens builds a graph that many steps deep and retains every intermediate. The decorator is the entire reason inference does not OOM. |
| `training/nanogpt/train.py:292` | `for micro_step in range(gradient_accumulation_steps):` — the accumulation loop. Note that nothing zeroes `.grad` inside it; that is the point. |
| `training/nanogpt/train.py:301` | `loss = loss / gradient_accumulation_steps`. The mean-of-means normalisation. Correct for equal micro-batches, biased for ragged ones. §2.6 and Exercise 3. |
| `training/nanogpt/train.py:305` | `scaler.scale(loss).backward()`. The traversal. Everything upstream of `loss` in the graph is consumed and freed as this runs — call it twice without `retain_graph=True` and you get "Trying to backward through the graph a second time." |
| `training/nanogpt/train.py:314` | `optimizer.zero_grad(set_to_none=True)` with the comment "no need for this memory anymore." A model-sized deallocation, deliberately placed immediately after `step()` rather than before the next forward. |
| `training/nanogpt/train.py:323` | `lossf = loss.item()` with the comment "this is a CPU-GPU sync point." Compare with how a production trainer avoids it: `training/olmo-core/src/olmo_core/train/trainer.py:1037` buffers metrics as unevaluated device tensors and drains them every N steps. |

### Detach discipline in production — `training/olmo-core/`

| Where | What to look at, and why |
|---|---|
| `training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:393` | `micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)`. Gradient accumulation as a *spatial* split sized by a memory budget, not a loop counter. The number of accumulation steps is derived from how much activation memory fits. |
| `.../train_module.py:414` | `ce_batch_loss += get_local_tensor(ce_loss.detach())`, followed by `del ce_loss` on the next line. The canonical fix for the accumulation leak in §2.7. The `del` is belt-and-braces: `.detach()` already broke the edge, the `del` drops the last Python reference. |
| `.../train_module.py:422` | `loss.backward()` — one backward per micro-batch, inside the loop. Compare to `_train_microbatch_context` (`:566`), which suppresses cross-rank gradient reduction on all but the last micro-batch. Design-only for us (`ASSUMPTIONS.md: single-device-only`) but the pattern is the one to copy. |

### The store-versus-recompute policy surface — `training/torchtitan/`

| Where | What to look at, and why |
|---|---|
| `training/torchtitan/torchtitan/distributed/activation_checkpoint.py:166` | `class FullAC` — recompute the entire transformer block during backward. The `√n` story of §2.5, at block granularity, in about eight lines. |
| `.../activation_checkpoint.py:185` | `class SelectiveAC` — per-*op* granularity: save the outputs of ops that are expensive to recompute, recompute the rest. Read `_get_default_save_ops` above it (line 31) for the actual list — SDPA variants, `linear`, `topk`, and the collectives. |
| `.../activation_checkpoint.py:271` | `if func in mm_ops and meta[mm_count_key] % 2 == 0: return CheckpointPolicy.PREFER_RECOMPUTE` — **recompute every second matmul.** That "2" is a hand-tuned constant with no derivation behind it. This single line is the honest state of the art in recompute policy, and it is the seed of §8. |
| `.../activation_checkpoint.py:290` | `class MemoryBudgetAC` — hand the whole decision to the compiler's partitioner under a memory budget. The existence of three coexisting policies in one file is the admission that nobody knows the rule. |
| `.../activation_checkpoint.py:122` | `preserve_rng_state: bool = True`. Recompute must reproduce the forward exactly, so dropout's RNG state has to be stashed and restored. This is the "recompute is exact" claim of §1 having to be *engineered*, not assumed — and it is a determinism hazard worth knowing about before the Hardware Validation Gate. |

---

## 5. Exercises

Run all of these from a scratch directory outside the repo — they are exploratory, not rig
code, and the house rule (`CLAUDE.md` → Engineering conventions) is that anything reused
migrates into the rig and acquires tests. Activate first:

```powershell
cd C:\projects\School\chiron
. .\scripts\activate-lab.ps1
```

**Standing hardware caveats for all three.** `[M]` Keep every single tensor under 31 GiB
(`ASSUMPTIONS.md: large-tensor-fault-32gib`) — a 32 GiB buffer hangs at 0 CPU with no
error, so a runaway shape does not crash, it stalls. `[C]` bf16 numerics on gfx1151 are
unproven (`bf16-numerics-unproven`), so **use fp32 for anything you intend to compare for
correctness** and bf16 only where you are measuring bandwidth. System RAM after the 96 GB
UMA carve-out is 31.6 GB `[M]`, so CPU fallbacks must be sized small — that is a real
constraint on this machine, not a courtesy.

---

### Exercise: stride forensics and the price of `.contiguous()`

**Difficulty 2/5. ~30 min to write, <1 min to run (GPU), ~2 min (CPU).**

Establish empirically that view/transpose are metadata-only and `.contiguous()` is not,
and put a GB/s number on the second one.

```python
# strides.py
import torch, time

def report(name, t):
    print(f"{name:26s} shape={tuple(t.shape)!s:22s} stride={t.stride()!s:26s} "
          f"contig={t.is_contiguous()!s:5s} ptr={t.data_ptr()}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| torch:", torch.__version__)

B, T, nh, hs = 64, 256, 6, 64
x = torch.randn(B, T, nh * hs, device=dev, dtype=torch.bfloat16)
report("x (B,T,C)", x)
v = x.view(B, T, nh, hs);  report(".view(B,T,nh,hs)", v)
h = v.transpose(1, 2);     report(".transpose(1,2)", h)
try:
    h.view(B, T, nh * hs)
    print("!! view succeeded — unexpected")
except RuntimeError as e:
    print("view refused:", str(e).splitlines()[0])
c = h.contiguous();        report(".contiguous()", c)

def bandwidth(t, iters=100):
    for _ in range(5):
        t.contiguous()                      # warm up allocator + kernels
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        t.contiguous()
    if dev == "cuda": torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    moved = 2 * t.numel() * t.element_size()          # one read + one write
    return dt, moved, moved / dt / 1e9

print(f"\n{'scale':>6} {'MiB moved':>10} {'ms':>8} {'GB/s':>8}")
for scale in (1, 4, 16):
    xb = torch.randn(B * scale, T, nh * hs, device=dev, dtype=torch.bfloat16)
    hb = xb.view(B * scale, T, nh, hs).transpose(1, 2)
    dt, moved, gbs = bandwidth(hb)
    print(f"{scale:>6} {moved/2**20:>10.1f} {dt*1e3:>8.3f} {gbs:>8.1f}")
    del xb, hb
```

**What you must be able to state afterwards.**

1. `x.data_ptr() == v.data_ptr() == h.data_ptr()`, and `c.data_ptr()` differs. Views are
   free; the copy is not.
2. The predicted single-layer traffic at scale 1 is `2 × 64 × 256 × 384 × 2 = 24.0 MiB`.
   Confirm the script's `MiB moved` column agrees.
3. Achieved GB/s versus the `[M]` 199.9 GB/s streaming-copy reference.

`[M]` **Reference run, 2026-07-26, gfx1151, torch `2.12.0a0+rocm7.13.0a20260313`, single
run** — printed by exactly the script above:

```
x (B,T,C)          shape=(64, 256, 384)    stride=(98304, 384, 1)       contig=True   ptr=12952010752
.view(B,T,nh,hs)   shape=(64, 256, 6, 64)  stride=(98304, 384, 64, 1)   contig=True   ptr=12952010752
.transpose(1,2)    shape=(64, 6, 256, 64)  stride=(98304, 64, 384, 1)   contig=False  ptr=12952010752
view refused: view size is not compatible with input tensor's size and stride ...
.contiguous()      shape=(64, 6, 256, 64)  stride=(98304, 16384, 64, 1) contig=True   ptr=12964593664

 scale  MiB moved       ms     GB/s
     1       24.0    0.133    188.7
     4       96.0    0.582    172.8
    16      384.0    2.182    184.6
```

**I predicted this would land well below the 199.9 GB/s streaming reference and it does
not** — 173–189 GB/s is 86–94% of it. The lesson is the one this lab keeps relearning: an
intuition about strided access ("gathers are slow") imported from a cache-hierarchy world
does not transfer to a GPU whose inner stride here is still 64 contiguous bf16 elements
(128 bytes, a full cache line and then some). The gather is coarse-grained enough to
stream. Had the transpose been on the *last* axis, the story would differ — and that is a
20-minute follow-up worth running before you trust any layout-cost estimate for Proteus.

Note also the stride tuples: they are exactly the ones derived by hand in §2.1, including
`(98304, 16384, 64, 1)` after `.contiguous()`. Deriving strides on paper and having the
machine agree is worth doing once.

**CPU fallback.** Set `dev = "cpu"` at the top. Everything works; expect roughly 10–40 GB/s
depending on how the copy vectorises. The *structural* result (views free, contiguous
bandwidth-bound, strided below peak) is identical, which is the point — layout is an
architecture-independent property.

**Extension if you have another 20 minutes.** Add `.permute(0, 2, 1, 3)` and
`.reshape(B, T, nh*hs)` and show that `reshape` silently succeeds where `view` threw, then
show `data_ptr()` changed. That is the API hiding a 24 MiB copy from you.

---

### Exercise: build an activation ledger and find the T² term

**Difficulty 3/5. ~60–90 min. Sweep runs in ~2 min (GPU), ~5 min (CPU).**

Build an itemised bill for what backward is holding, using
`torch.autograd.graph.saved_tensors_hooks`, then measure whether the quadratic attention
term exists on the fused and manual paths. This is a 20-line activation profiler and it is
the most reusable thing in this module.

Two design decisions in the code below are the whole exercise, and I got both wrong on the
first attempt:

- **Exclude parameters.** `X̄ = Ȳ Wᵀ` means every `nn.Linear` saves its *weight* for
  backward. Those are parameters — already resident, not a new cost — but the hook sees
  them and they dominated my first ledger at 40.6 MiB against 165 MiB of real activations.
  Filter by `data_ptr` against `m.parameters()`.
- **Deduplicate by `data_ptr` alone, not by `(data_ptr, shape)`.** The same softmax output
  is saved once as `(16, 6, 256, 256)` and again as `(96, 256, 256)` — a reshaped view for
  the batched matmul. Keying on shape counted it twice and doubled the fitted quadratic
  coefficient to exactly the wrong plausible answer.

```python
# activation_ledger.py
import sys, collections, torch, numpy as np
sys.path.insert(0, r"C:\projects\School\chiron\research\reference\training\nanogpt")
from model import GPT, GPTConfig

dev = "cuda" if torch.cuda.is_available() else "cpu"
B = 16 if dev == "cuda" else 2          # CPU: keep it small, 31.6 GB system RAM

def measure(T, flash):
    torch.manual_seed(1337)
    cfg = GPTConfig(block_size=T, vocab_size=65, n_layer=6, n_head=6,
                    n_embd=384, dropout=0.0, bias=True)   # dropout off: cleaner ledger
    m = GPT(cfg).to(dev).train()
    for blk in m.transformer.h:
        blk.attn.flash = flash
        if not flash:
            blk.attn.register_buffer(
                "bias", torch.tril(torch.ones(T, T, device=dev)).view(1, 1, T, T))

    param_ptrs = {p.data_ptr() for p in m.parameters()}
    seen, act = set(), collections.Counter()
    def pack(t):
        ptr = t.data_ptr()
        if ptr in seen or ptr in param_ptrs:      # dedup, and drop parameters
            return t
        seen.add(ptr)
        act[(tuple(t.shape), str(t.dtype))] += t.numel() * t.element_size()
        return t
    def unpack(t):
        return t

    idx = torch.randint(0, 65, (B, T), device=dev)
    if dev == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        _, loss = m(idx, idx)
    peak = torch.cuda.max_memory_allocated() if dev == "cuda" else 0
    loss.backward()
    return act, sum(act.values()), peak

print(f"device={dev} B={B}")
print("sdpa backends: flash=%s memeff=%s math=%s" % (
    torch.backends.cuda.flash_sdp_enabled(),
    torch.backends.cuda.mem_efficient_sdp_enabled(),
    torch.backends.cuda.math_sdp_enabled()))
print(f"{'T':>6} {'flash':>6} {'act MiB':>10} {'peak MiB':>10}")
rows = {}
for flash in (True, False):
    for T in (64, 128, 256):
        act, total, peak = measure(T, flash)
        rows[(flash, T)] = total
        print(f"{T:>6} {str(flash):>6} {total/2**20:>10.2f} {peak/2**20:>10.1f}")

for tag, fl in (("SDPA", True), ("manual", False)):
    act, _, _ = measure(256, flash=fl)
    print(f"\nTop activation tensors, {tag}, T=256 (summed over 6 layers):")
    for (shape, dt), b in act.most_common(5):
        print(f"  {str(shape):>26s} {dt:>16s} {b/2**20:>9.2f} MiB")

# three-point fit  saved(T) = k + a*T + c*T^2
for flash in (True, False):
    Ts = np.array([64.0, 128.0, 256.0])
    ys = np.array([rows[(flash, t)] for t in (64, 128, 256)], dtype=float)
    A = np.stack([np.ones_like(Ts), Ts, Ts**2], axis=1)
    k, a, c = np.linalg.solve(A, ys)
    print(f"flash={flash}: k={k:,.0f} B  a={a:,.0f} B/tok  c={c:,.1f} B/tok^2")
print("predicted c for ONE saved (B,nh,T,T) fp32 per layer:", 6*B*6*4)
```

Run it twice: once as-is, and once with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` set in
the environment.

**What you must be able to state afterwards.**

1. The predicted quadratic coefficient for **one** saved `(B, nh, T, T)` fp32 tensor per
   layer is `L × B × nh × 4 = 6 × 16 × 6 × 4 = 2,304` bytes per `T²`. Confirm the manual
   path fits that, not `4,608` — autograd saves the softmax *output* once and reuses the
   same storage for both the softmax backward and the `att @ v` backward. Deriving 2,304
   and 4,608 as the two candidate answers, then letting the measurement choose, is the
   point of the exercise.
2. **Whether the SDPA path differs from the manual path at all.** On this machine, by
   default, it does not — see the `[M]` table in §2.4. With the AOTriton variable set, its
   `c` goes to exactly 0.0. Report both.
3. The gap between the ledger total and `max_memory_allocated()`. They are different
   numbers: `peak` includes transient workspace that was allocated and freed inside the
   forward, plus the parameters your ledger deliberately excluded. Explain the sign.
4. Re-run once with `dropout=0.2` (the shipped `train_shakespeare_char.py:25` value) and
   find the bool masks in the itemised bill at one byte per element — the `+1` in
   Korthikanti's `5 = 2 + 2 + 1` (§2.5), observed rather than quoted.

`[M]` **Reference run, 2026-07-26, gfx1151, `B=16`, fp32, single run:**

```
                        act MiB @ T=64 / 128 / 256      fitted c (B/tok^2)
manual                    156.39 / 330.82 / 733.83            2,310.0
SDPA, default env         156.36 / 330.73 / 733.46            2,304.0
SDPA, AOTRITON=1          147.50 / 295.01 / 590.02                0.0
```

**Sizing check before you run.** Largest case, `T=256`, `B=16`, fp32: ~734 MiB of
activations, ~850 MiB allocator peak. Far inside the `[M]` ≥62 GiB fast tier and nowhere
near the 31 GiB per-tensor fault. If you extend the sweep to `T=1024` at `B=16`, the score
tensor alone is `16 × 6 × 1024² × 4 = 402 MiB` per layer — still fine, but do the
arithmetic before you type the number, not after.

**CPU fallback.** Set `B = 2`. Everything works except `max_memory_allocated`, which reads
0; the ledger is the primary instrument anyway, which is deliberate — it is device-agnostic
and therefore the better tool. Predicted `c` on CPU at `B=2` is `6 × 2 × 6 × 4 = 288`.

**Caveat to carry.** Deduplicating by `data_ptr` alone now *under*-counts in the opposite
direction: two genuinely different tensors carved from one allocator block can share a
base pointer in some paths. The profiler is approximate. Write down that it is, and by
roughly how much, before you use it to make a decision — a profiler you trust blindly is
worse than none.

---

### Exercise: prove the gradient-accumulation identity, then break it

**Difficulty 2/5. ~45 min. Runs in <1 min on CPU.**

CPU fp32 is the **reference** here, not the fallback — `bf16-numerics-unproven` means the
GPU is the thing under test. Run CPU first, then GPU, and compare.

```python
# accumulation_identity.py
import sys, torch
sys.path.insert(0, r"C:\projects\School\chiron\research\reference\training\nanogpt")
from model import GPT, GPTConfig

def run(dev):
    torch.manual_seed(1337)
    cfg = GPTConfig(block_size=64, vocab_size=65, n_layer=2, n_head=2,
                    n_embd=64, dropout=0.0, bias=True)
    m = GPT(cfg).to(dev).train()
    X = torch.randint(0, 65, (32, 64), device=dev)
    Y = torch.randint(0, 65, (32, 64), device=dev)

    def grads():
        return {n: p.grad.detach().clone()
                for n, p in m.named_parameters() if p.grad is not None}

    def maxdiff(a, b):
        return max((a[k] - b[k]).abs().max().item() for k in a)

    # reference: one batch of 32, mean CE over all 32*64 tokens
    m.zero_grad(set_to_none=True)
    _, loss = m(X, Y); loss.backward()
    ref = grads()
    scale = max(v.abs().max().item() for v in ref.values())

    def accumulate(splits, weighted):
        m.zero_grad(set_to_none=True)
        total = sum(splits) * 64            # total tokens in the batch
        i = 0
        for n in splits:
            _, l = m(X[i:i+n], Y[i:i+n])
            (l * (n * 64) / total if weighted else l / len(splits)).backward()
            i += n
        return grads()

    print(f"\n--- {dev} --- reference |grad|_max = {scale:.6f}")
    print(f"equal splits, mean-of-means   max|d| = {maxdiff(ref, accumulate([8]*4, False)):.3e}")
    print(f"equal splits, token-weighted  max|d| = {maxdiff(ref, accumulate([8]*4, True)):.3e}")
    print(f"RAGGED,       mean-of-means   max|d| = {maxdiff(ref, accumulate([5,9,7,11], False)):.3e}")
    print(f"RAGGED,       token-weighted  max|d| = {maxdiff(ref, accumulate([5,9,7,11], True)):.3e}")

    # forgetting zero_grad
    m.zero_grad(set_to_none=True)
    for _ in range(2):
        _, l = m(X, Y); l.backward()
    doubled = grads()
    print(f"two backwards, no zero_grad   max|d| = {maxdiff(ref, doubled):.3e}"
          f"   (expect ~= reference |grad|_max)")
    return ref

ref_cpu = run("cpu")
if torch.cuda.is_available():
    ref_gpu = run("cuda")
    d = max((ref_cpu[k] - ref_gpu[k].cpu()).abs().max().item() for k in ref_cpu)
    print(f"\nCPU vs gfx1151, fp32, same seed: max|d| = {d:.3e}")
```

**What you must be able to state afterwards.**

1. Equal splits, mean-of-means: `max|Δ|` at fp32 round-off, order `1e-7`. The identity
   holds. Non-zero-ness is accumulation-order noise, not a bug.
2. **Ragged splits, mean-of-means: materially non-zero.** Quantify it as a fraction of
   `|grad|_max`. This is nanoGPT's `train.py:301` formulation failing, on purpose.
3. Ragged splits, token-weighted: back to round-off. This is OLMo-core's formulation
   (`train_module.py` and the `loss_div_factor` note in `CODE_MAP.md`) being exactly right
   for the ragged case.
4. Two backwards without `zero_grad`: `max|Δ|` ≈ the reference gradient magnitude, because
   the result is exactly 2× the reference. **No exception was raised.** Sit with that.
5. The CPU-vs-gfx1151 fp32 difference. If it exceeds ~1e-4 relative, that is a numerics
   signal on our instrument and belongs in `ASSUMPTIONS.md` against
   `bf16-numerics-unproven` as partial evidence — with the caveat that this is fp32, so a
   failure here would be worse news than a bf16 failure.

`[M]` **Reference run, 2026-07-26, single run, seed 1337, reference `|grad|_max = 0.034949`:**

| Case | CPU `max|Δ|` | gfx1151 `max|Δ|` |
|---|---|---|
| equal splits, mean-of-means | 5.588e-09 | 1.583e-08 |
| equal splits, token-weighted | 5.588e-09 | 1.583e-08 |
| **ragged splits, mean-of-means** | **1.238e-02** | **1.238e-02** |
| ragged splits, token-weighted | 6.403e-09 | 1.630e-08 |
| two backwards, no `zero_grad` | 3.495e-02 | 3.495e-02 |
| CPU vs gfx1151, fp32, same seed | — | 3.912e-08 |

Read the third row carefully. `1.238e-02` against a largest-gradient magnitude of
`3.495e-02` is a **35% error on the biggest component of the gradient**, from nothing but
uneven micro-batch sizes and a `/ M` where a `/ total_tokens` belonged. This is not a
rounding artefact you can wave at; it is a different optimisation problem. Any Proteus arm
that packs documents, masks labels, or uses variable-length sequences and normalises the
way `train.py:301` does is silently training on a reweighted objective.

The last row is the useful by-product: fp32 gradients on gfx1151 agree with CPU to
`3.9e-8` absolute, `~1.1e-6` relative to `|grad|_max`. That is consistent with reassociated
fp32 summation and nothing worse. It is **one seed on one tiny model** — an anecdote by the
house standard — but it is the first gradient-correctness evidence this lab has on this
machine, and it cost under a minute.

**GPU note.** This model is 2 layers / 64 wide; it will not stress anything. Run it on GPU
purely for the cross-device comparison in step 5. A single-run comparison is an anecdote by
the house standard — if the number is interesting, re-run with three seeds before writing
it anywhere.

---

## 6. Self-check

Answers at the end of this file.

1. A float32 tensor has `shape = (4, 8, 16)` and `stride = (128, 16, 1)`. Is it contiguous?
   What is the byte offset of element `[2, 3, 4]`? After `.transpose(0, 2)`, what are the
   shape and strides, and will `.view(512)` succeed?

2. `x` has shape `(1, 1000)` and `requires_grad=True`. You compute
   `y = x.expand(1000, 1000)` and then `y.sum().backward()`. How many bytes does `y`
   occupy? What is the shape of `x.grad`, and what is in it?

3. You call `loss.backward()` and then call it again without `retain_graph=True`. PyTorch
   raises. Name the resource that was released between the two calls, and explain why the
   framework releases it eagerly instead of waiting for garbage collection.

4. In a micro-batch loop, `running += loss` (no `.detach()`) instead of
   `running += loss.detach()`. Peak activation memory grows by what factor over `M`
   micro-batches, and why does the *gradient* nevertheless come out correct?

5. FlashAttention removes the `O(T²)` term from activation memory. Does it remove the
   `O(T²)` term from the FLOP count? State precisely what is being traded for what.

6. `[M]` Our machine hangs on single tensors ≥32 GiB. For a materialised bf16 attention
   score tensor of shape `(B, nh, T, T)` with `nh = 8`, at what `(B, T)` pairs do you hit
   that? Show the arithmetic, and name the one code change that makes the question moot.

---

## 7. What is still unsolved here

**There is no closed form for activation memory, and this is not a gap that is closing.**
Compare with the KV cache, where `research/memory/kv-cache-mechanics.md` derives
`2 · L · n_kv · d_h · b` as an *exact* function of five config fields. Activation memory has
no equivalent, because it depends on which kernels fused, which ops the autograd
implementation chose to save inputs versus outputs for, what `torch.compile`'s partitioner
decided, and which recompute policy is active. Korthikanti's `s·b·h·(34 + 5as/h)` `[C]`
(`2205.05198`, May 2022) is the most-cited attempt and it predates fused attention by
months. `[A]` High confidence that no maintained successor exists; the cheapest test that
would move it is a literature search for a 2025–26 formula that explicitly covers SDPA and
selective recompute, and I did not find one. **Practical consequence for this lab: you
measure activation memory, you do not compute it** — which is why Exercise 2 is a profiler
and not a spreadsheet.

**Recompute policy selection is heuristic and everybody knows it.** `torchtitan`'s
production answer is "save every second matmul"
(`training/torchtitan/torchtitan/distributed/activation_checkpoint.py:271`). That `2` has
no derivation. The same file ships three mutually exclusive policies —
`FullAC`, `SelectiveAC`, `MemoryBudgetAC` (lines 166, 185, 290) — and the third one exists
precisely to hand the decision to a compiler partitioner because no analytic rule is known.
`[C]` Active work continues here (e.g. Adacc, arXiv `2508.00806`, unifying compression and
recomputation adaptively — id seen in search results 2026-07-26, abstract not read in full;
and PyTorch's own survey of SAC and Memory Budget APIs at
https://pytorch.org/blog/activation-checkpointing-techniques/). Treat "which activations to
keep" as an **open optimisation problem with a known objective and no known solution**.

**Whether activation checkpointing and KV eviction are the same problem is genuinely
open, and it is the bridge from this module into the memory track.** `[C]` `2607.08032`
(Jul 2026, "What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction in
LLMs and Agents") argues that KV-cache eviction, prompt pruning, recurrent-state bounding
and agent-memory consolidation are one rate–distortion problem under a resource budget. It
does **not** cover activation checkpointing, and the reason it plausibly should not is the
distinction `research/memory/README.md` identifies: activations are *exactly*
reconstructible, so there is no distortion term at all — only a compute cost. That would
make activation checkpointing the degenerate, zero-distortion corner of the same
framework. `[A]` Medium confidence that this framing is correct and unpublished; the
cheapest test is reading `2607.08032` closely enough to see whether their formalism admits
a zero-distortion limit. If it does, it is a small, real, writable contribution.

**Whether the memory-efficient attention backend on gfx1151 is trustworthy is unknown, and
we now know it is not optional.** `[M]` §2.4 establishes that without
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` this wheel materialises the full `(B, nh, T, T)`
score tensor even when you call `scaled_dot_product_attention`, which puts every
long-context experiment on the wrong side of the `[M]` 32 GiB single-tensor fault. So the
choice is between a known-bad memory profile and a backend AMD itself labels experimental.
`[A]` Low confidence in either direction on numerics; the cheapest discriminating test is
an attention-output comparison (fp32 and bf16) between the math and AOTriton backends at
the head shapes we intend to run, which is one afternoon and is now a named Hardware
Validation Gate item rather than a vague "verify the AOTriton speedup" note. Nothing about
this is in the literature — it is a property of one wheel on one chip — which is precisely
why it has to be measured rather than assumed.

**Backward-pass determinism is not free, and on our hardware it is not established.**
Several backward kernels (embedding backward, scatter-add, index_put) use atomic
accumulation, so summation order varies run to run and results differ in the last bits.
`torch.use_deterministic_algorithms(True)` covers many of these on CUDA; coverage on ROCm
gfx1151 is unknown to us. Layered on top: `ASSUMPTIONS.md: bf16-numerics-unproven` is
`untested` and cites five documented gfx1151 bf16 bugs `[C]` (ROCm #6034). **We cannot
currently assert that a gradient computed on this machine matches an fp32 reference.**
That is a Hardware Validation Gate item, and Exercise 3 step 5 is the cheapest partial
attack on it.

**Activation offload to host memory has no cost model on a unified-memory machine.** The
standard analysis — offload activations to CPU DRAM, pay PCIe, win capacity — assumes a
discrete GPU across a bus. On the Z13 the "host" and "device" are the same physical DRAM,
partitioned by a BIOS carve-out. `[M]` We measured that the carve-out controls a fast tier
(≥62 GiB at ~200 GB/s) and that beyond the old boundary bandwidth fell to 61–114 GB/s
(`notebook/uma-carveout-controls-fast-tier.md`, single run per arm). So "offload" here is
not a bus crossing, it is a **move between two bandwidth tiers of one memory**, and whether
it pays is an open, cheap, locally answerable question. `[A]` Low confidence in either
direction. This is an available experiment that nobody else can easily run, which by the
G3 information-per-dollar test makes it interesting.

**Higher-order gradients have no good memory story.** Double-backward (needed for
gradient-penalty regularisers, some meta-learning, and influence functions) retains the
forward graph *and* the backward graph. Nothing in the recompute literature addresses it
well. Out of scope for Proteus today; flagged so that a future "let's just add a gradient
penalty" is costed rather than assumed.

---

## Answers to the self-check

**1.** Contiguous strides for `(4, 8, 16)` are the suffix products: `(8·16, 16, 1) =
(128, 16, 1)`. They match, so **yes, contiguous.** Element `[2,3,4]` is at
`2·128 + 3·16 + 4·1 = 308` elements → **byte 1232** (float32, 4 bytes). After
`.transpose(0, 2)` the shape is `(16, 8, 4)` and the strides are the same tuple permuted:
`(1, 16, 128)`. `.view(512)` **fails** — walking that tensor in logical order visits flat
offsets `0, 16, 32, …, 112, 1, 17, …`, which is not an arithmetic progression, so no single
stride describes it. `.reshape(512)` would succeed by copying all 2048 bytes.

**2.** `y` occupies **0 additional bytes of data**. `expand` sets `stride[0] = 0`; the
descriptor is a few dozen bytes of metadata over `x`'s existing 4000-byte storage. `x.grad`
has shape **`(1, 1000)`** — the same shape as `x`, always — and every entry is **1000.0**,
because each element of `x` was read by 1000 output positions and `sum()` sends a gradient
of 1.0 to each. This is §2.2's adjoint-of-broadcast-is-a-sum, and it is also the clean
demonstration that a free forward broadcast becomes a real reduction in backward.

**3.** The **saved forward tensors** (the activations referenced by the graph's nodes) are
released as `backward()` consumes each node; the graph nodes themselves are then
unreferenced and freed. The framework releases eagerly because those tensors are the single
largest live allocation in a training step — §2.5 puts them an order of magnitude above the
model at nanoGPT scale — and waiting for Python's refcount to drop, let alone a GC cycle, would mean peak
memory equal to the sum of forward and backward working sets rather than their overlap.
`retain_graph=True` opts out and pays that cost.

**4.** Peak activation memory grows by a factor of **M**: `running` transitively pins every
micro-batch's graph, so instead of one micro-batch's activations being live at a time, all
`M` are. The **gradient is still correct**, which is what makes this bug so durable —
`.grad` accumulated correctly across the `M` backward calls regardless of what else held
references. You find it by OOM at some `M`, never by a wrong number. The production fix is
`training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:414`.

**5.** **No.** FlashAttention still performs `O(T²)` work — every query still attends to
every key. What it removes is the *materialisation* of the `T×T` intermediate: scores are
computed, softmaxed, and consumed tile-by-tile in on-chip memory, and only the output plus
a per-row log-sum-exp statistic (`O(T)`) leaves the kernel. The trade is **memory traffic
and memory capacity for redundant compute** — the backward pass recomputes each score tile
from Q and K rather than reading it back. Because the recompute happens in registers/SRAM
while the alternative was a round trip to HBM, it is usually *faster* as well as smaller,
which is why it is not presented as a tradeoff in practice. It is still a tradeoff in
principle, and it is precisely the same store-versus-recompute decision as activation
checkpointing (§2.5), applied inside one kernel.

**6.** Bytes `= B · nh · T² · 2`. With `nh = 8`:

```
B=1, T=32,768 :  1 × 8 × 1,073,741,824 × 2 = 17,179,869,184 B = 16.0 GiB   → safe
B=2, T=32,768 :                              34,359,738,368 B = 32.0 GiB   → the fault
B=1, solve for the 32 GiB threshold:
        T² = 32 · 2³⁰ / (8 · 2) = 2,147,483,648  →  T = 46,341 tokens
B=4, T=16,384 :  4 × 8 × 268,435,456 × 2   = 17,179,869,184 B = 16.0 GiB   → safe
B=8, T=16,384 :                              34,359,738,368 B = 32.0 GiB   → the fault
```

Note the invariant: the fault surface is the curve `B · T² = 2³¹` at `nh = 8`, bf16. The
change that makes it moot is a **genuinely memory-efficient attention kernel**, which never
allocates the `(B, nh, T, T)` tensor at all.

The trap — and this is the part worth having got wrong — is that calling
`scaled_dot_product_attention` (`model.py:64`) is **not sufficient on this machine.** `[M]`
§2.4 measures the default SDPA path on gfx1151 producing exactly the same quadratic
coefficient as the hand-written path, because PyTorch dispatches to the math backend while
still reporting `flash_sdp_enabled() == True`. You must also set
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, and then verify the numerics, because the
backend is experimental. If you answered "just use SDPA," you gave the answer that is
correct on NVIDIA and wrong here — which is the single most useful thing this module can
teach about working on unproven hardware.

---

## Sources

**arXiv, verified by fetching the abstract page on 2026-07-26:**

- `1502.05767` — *Automatic differentiation in machine learning: a survey*, Baydin,
  Pearlmutter, Radul, Siskind (2015-02-20, rev. 2018-02-05, JMLR 2018). The reverse-mode
  cost argument in §2.3.
- `1604.06174` — *Training Deep Nets with Sublinear Memory Cost*, Chen, Xu, Zhang, Guestrin
  (2016-04-21). The `√n` checkpointing result in §2.5.
- `2205.05198` — *Reducing Activation Recomputation in Large Transformer Models*,
  Korthikanti et al. (2022-05-10). The `s·b·h·(34 + 5as/h)` accounting.

**arXiv, from `research/reference/papers/README.md` (API-verified in that pass):**

- `2607.08032` — *What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction
  in LLMs and Agents* (2026). Cited in §7 for the unification claim and for what it does
  not cover.

**arXiv id seen in live search results 2026-07-26, abstract not read in full — flagged as
such rather than presented as read:**

- `2508.00806` — *Adacc: An Adaptive Framework Unifying Compression and Activation
  Recomputation for LLM Training*.

**Non-arXiv:**

- PyTorch engineering blog, *Current and New Activation Checkpointing Techniques in
  PyTorch* — https://pytorch.org/blog/activation-checkpointing-techniques/ (Selective
  Activation Checkpoint and the Memory Budget API; the framing that policy selection is a
  tuning problem).

**Local code, line numbers pinned to the revisions in `research/reference/PROVENANCE.md`:**

- `training/nanogpt/model.py:57`, `:64`, `:68`, `:72`, `:138`, `:159`, `:179`, `:187`,
  `:267`, `:305`
- `training/nanogpt/train.py:215`, `:292`, `:301`, `:305`, `:314`, `:323`
- `training/nanogpt/config/train_shakespeare_char.py:22`, `:25`
- `training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:393`,
  `:414`, `:422`
- `training/olmo-core/src/olmo_core/train/trainer.py:1037`
- `training/torchtitan/torchtitan/distributed/activation_checkpoint.py:122`, `:166`,
  `:185`, `:271`, `:290`
- `research/reference/CODE_MAP.md` — the OLMo-core "Worth knowing" note on
  `loss_div_factor`, and the nanoGPT section on the hardcoded `flops_promised = 312e12`.

**Measurements taken while writing this module, 2026-07-26** (gfx1151, native Windows,
torch `2.12.0a0+rocm7.13.0a20260313`, HIP 7.2.0, lab venv `C:\venvs\lab` via
`scripts/activate-lab.ps1`). All are **single runs — anecdotes by the house standard**
(`CLAUDE.md`: ≥3 seeds, confidence intervals). They are reported because two of them
contradicted what this module originally asserted, and because the effect sizes are large
and the boundaries sharp. **Three of them should become rows or notebook entries and have
not yet been:**

1. **SDPA on gfx1151 does not take a memory-efficient path by default.** Fitted quadratic
   activation coefficient identical to the manual attention path (2,304 vs 2,310
   B/token²); goes to exactly 0.0 with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`.
   Activations at `T=256, B=16` fall 733.8 → 590.0 MiB. **Candidate `ASSUMPTIONS.md` row;
   affects every long-context arm and interacts with `large-tensor-fault-32gib`.**
2. **A strided head-layout `.contiguous()` reaches 173–189 GB/s**, 86–94% of the `[M]`
   199.9 GB/s streaming-copy reference — *not* "well below" it, as this module's first
   draft predicted. The prediction and its refutation are both left in §5 on purpose.
3. **fp32 gradients on gfx1151 match CPU to 3.9e-8 absolute** (~1.1e-6 relative to
   `|grad|_max`) on a 2-layer 64-wide GPT, one seed. First gradient-correctness evidence
   on this machine; partial, cheap, and relevant to `bf16-numerics-unproven`.
4. **Mean-of-means micro-batch normalisation costs 35% of the largest gradient component**
   on ragged splits `[5,9,7,11]`, and token-weighted normalisation recovers exactness to
   round-off. Not hardware-specific; a property of `train.py:301`'s formulation.

**Lab measurements (`[M]`) from prior work:**

- `ASSUMPTIONS.md` rows `gpu-fast-tier-size` (≥62 GiB at ~200 GB/s),
  `large-tensor-fault-32gib` (31 GiB clean at 199.9 GB/s; 32 GiB hangs at 0 CPU),
  `bf16-numerics-unproven` (untested), `torch-build`
  (`2.12.0a0+rocm7.13.0a20260313`, HIP 7.2.0).
- `notebook/uma-carveout-controls-fast-tier.md` — the bandwidth sweep and the ≥32 GiB
  fault. Single run per arm; an anecdote by the house standard, reported because the effect
  sizes are large and the boundaries sharp.
- `ENVIRONMENT.md` — 31.6 GB system RAM after the 96 GB UMA carve-out, which is what sizes
  the CPU fallbacks.

**Consistency note.** This module is written to agree with `research/memory/README.md`
(reconstructibility as the partitioning axis), `research/memory/kv-cache-mechanics.md`
(the exact KV formula, contrasted in §7 with the absence of one for activations), and
`research/memory/open-problems-ranked.md` (per-layer allocation as the consequence of the
32 GiB fault). Where it goes beyond them — the claim that activation checkpointing is the
zero-distortion corner of the compaction problem — it is tagged `[A]` and carries its own
cheapest test.
