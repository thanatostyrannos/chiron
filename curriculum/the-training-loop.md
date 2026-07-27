---
title: The training loop — batch, forward, loss, backward, step, and where every byte lives
version: 1.0.0
date: 2026-07-26
owner: curriculum-author
track: A — Foundations
prereqs: tensors-and-autograd, transformer-forward-pass-by-hand
difficulty: 2/5 conceptually, 3/5 in the details that bite
time: ~2 h reading, ~3 h exercises (splits cleanly across two evenings)
---

# The training loop

## 1. What this module settles

Six statements — fetch a batch, run forward, compute a loss, run backward, step the
optimizer, zero the gradients — are the entire loop, and everything hard about training
is a detail hiding inside one of them. This module makes you able to say, for a given
config, exactly how many bytes of each kind exist at each point in the step and which
term grows with what; to state which gradient-accumulation normalization is correct and
by how much the common one is wrong; and to say precisely why the field standardized on
bf16 rather than fp16, what bf16 costs, and what a fixed seed does and does not buy you.
It deliberately does **not** go deep on any single statement. Cross-entropy, perplexity,
AdamW's failure modes, LR schedules and clipping get their full treatment in
`loss-and-optimization.md`; the `6·N·D` derivation and MFU conventions get theirs in
`scaling-laws-and-flops-budget.md`; optimizer *choice* and distributed training are
`research/notes/pretraining-recipes.md` and Track D. All of those declare this module as a
prerequisite, because they are elaborations of one statement in a loop you have to be able to
see whole first.

---

## 2. Theory in plain language

### 2.1 The loop, in six statements

```python
for step in range(max_steps):
    x, y = next(loader)                 # 1. fetch
    logits = model(x)                   # 2. forward
    loss = cross_entropy(logits, y)     # 3. loss
    loss.backward()                     # 4. backward
    optimizer.step()                    # 5. update
    optimizer.zero_grad()               # 6. reset
```

That is it. nanoGPT's version is 23 lines (`training/nanogpt/train.py:255`–`:314`) and
OLMo-core's production version is the same six statements split across a scheduler and a
device-side module (`trainer.py:1466`, `train_module.py:345`). Nothing in a 15-trillion-token
frontier run changes the shape; what changes is what fills each statement.

The thing worth internalizing first is that **this is a fixed-point iteration on a
parameter vector, driven by a stream, with no notion of correctness at any single step.**
Step 500 being wrong does not throw. It does not corrupt an invariant. It just makes the
final artifact slightly worse in a way nothing downstream can detect. Contrast a
replication pipeline: a lost write eventually surfaces as a read that returns the wrong
value, and you have a consistency check that can find it. Here there is no read path and
no invariant. The only integrity check available is "does the loss curve continue," which
is a smoke alarm, not a checksum.

### 2.2 What this replaced

The loop above is the *stochastic* gradient method: estimate the gradient of the loss over
the whole dataset by evaluating it on a small random sample, and take a step. It replaced
full-batch gradient descent (evaluate on everything, step once — correct and unaffordable)
and second-order methods (build a curvature matrix — better steps, quadratic memory in
parameter count).

The optimizer inside step 5 has itself had one replacement that matters. Plain SGD uses a
single global learning rate; parameters with tiny gradients barely move. Adam `[C]`
(arXiv 1412.6980, Dec 2014) gives each parameter its own effective step size by dividing
by a running estimate of that parameter's gradient magnitude. AdamW `[C]`
(arXiv 1711.05101, Nov 2017) fixed a bug in how Adam applied weight decay — see §3.3 for
exactly what the bug was. AdamW has been the default since roughly GPT-3, and whether
matrix-preconditioned optimizers (Muon, SOAP) have displaced it is genuinely contested and
covered in `research/notes/pretraining-recipes.md` §1, not here.

### 2.3 The bridges, and where each one breaks

This is the part to read slowly. Every row's right-hand column is the actual content.

| You already own | Its counterpart in the loop | Where the analogy breaks |
|---|---|---|
| A batch job with checkpoint/restart | A training run | It never fails loudly. It returns a number, and the number is wrong by an amount you cannot bound from inside the job. |
| A write buffer flushed at commit | `.grad`, flushed by `optimizer.step()` + `zero_grad()` | Forgetting the flush raises nothing. `backward()` **adds into** `.grad`; skip the reset and by step *k* you are stepping on a sum of *k* gradients — an effective learning rate of *k·η* — and it looks like a diverging run, not a bug. |
| A WAL replayed to rebuild state | The autograd tape, replayed backwards | The tape is *consumed* on replay, it computes a different quantity (the gradient, not the state), and it pins every saved activation alive until traversed. An unreleased tape is a memory leak, not a durable artifact. |
| A stateful stream processor with per-key state | AdamW's `m` and `v`, one pair per parameter | The state is not reconstructible from the input stream. Lose the optimizer moments and you cannot replay to recover them — only re-train. |
| Group commit / write combining | Gradient accumulation; also metric batching | No correctness fallback. Get the flush point wrong and it silently trains on the wrong gradient. That is why OLMo-core makes it a context manager (`train_module.py:566`) rather than a flag. |
| Idempotent replay, exactly-once | Seeded determinism | Float addition is not associative and the hardware deliberately reorders reductions for throughput. You cannot make the op commutative. You can only pin the reduction order and pay for it, or accept bitwise drift. |
| Lossless compression on a hot tier | bf16 as the compute dtype | It is *lossy*, and the loss is a relative-error floor, not a bounded absolute error. The damage is rarely in the stored value; it is in what happens when you accumulate small updates into it (§3.6). |
| Hot/cold tiering with spill-to-slower-media | Activation checkpointing | The "spill" is not a write to a slower tier — it is discarding the data and recomputing it. The cost is FLOPs, not I/O, and the recomputed value is not guaranteed bit-identical to the original. |
| Logging is approximately free | Training telemetry | Reading a metric forces a host-device sync and stalls the pipeline. Observability costs throughput directly. OLMo-core arms a tripwire for exactly this: `trainer.py:1517` turns on CUDA sync-debug `warn` during training and `train/utils.py:137` suppresses it at the one place a sync is legitimate. |

One structural fact worth stating early because it defines the boundary of this lab's main
interest: **there is no KV cache during training.** K and V are recomputed from the residual
stream on every forward pass and discarded (`training/nanogpt/model.py:56`). Mnemosyne's
surface — eviction, tiering, prefix reuse — is exercised at inference time. The memory you
budget during training is *activations*, which is a different object with a different
lifetime and different economics.

---

## 3. The math that actually matters

### 3.1 The loss, and the one derivative you should know by heart

The model emits, for each position, a vector of **logits** `z ∈ ℝ^V` — one real number per
vocabulary entry, unnormalized. `V` is the vocabulary size. Softmax turns them into a
probability distribution:

```
p_i = exp(z_i) / Σ_j exp(z_j)
```

- `p_i` — the model's predicted probability for vocabulary entry `i`
- `z_i` — the logit for entry `i`
- `Σ_j` — sum over all `V` entries; the denominator is the *partition function*, the thing
  that makes the vector sum to 1

Cross-entropy loss for one position whose correct next token is `t`:

```
ℓ = −log p_t
```

Read that as: the number of **nats** of surprise the model assigned to the token that
actually occurred. Divide by `ln 2 ≈ 0.6931` to get bits. A model that assigns probability
1 to the right token has loss 0; one that assigns 1/V has loss `ln V`.

**Sanity check you should run on every new config:** an untrained model outputs roughly
uniform logits, so its first loss should be `ln V`. For nanoGPT's shakespeare_char, `V = 65`
and `ln 65 = 4.174`. For a GPT-2 vocab, `V = 50,304` and `ln 50,304 = 10.826`. If step 0
does not land near `ln V`, your initialization, your tokenizer, or your label alignment is
wrong, and you have found it in one second instead of one day.

Now the derivative, which is the single most load-bearing three symbols in the module:

```
∂ℓ/∂z_i = p_i − 1[i = t]
```

- `1[i = t]` — indicator: 1 if `i` is the correct token, 0 otherwise

So the gradient at the logits is *predicted distribution minus one-hot truth*. Nothing
else. Every gradient in the network is this vector pushed backwards through the layers.
Two consequences worth carrying:

1. The gradient magnitude at the output is bounded by 1 per component and shrinks as the
   model gets confident (`p_t → 1` makes the correct-class component → 0). Late in training,
   the signal is small — which is where finite precision starts to bite (§3.6).
2. Total gradient mass over the vocabulary is exactly 0 (`Σ_i p_i = 1`, and the one-hot sums
   to 1). Confidence is redistributed, never created.

For nanoGPT this is one line: `training/nanogpt/model.py:187`.

The published pass/fail number for the reference run is **1.4697** nats
(`training/nanogpt/README.md:51`), which is `1.4697 / 0.6931 = 2.120 bits per character`.
That is an information-theoretic quantity, not a latency SLO — and because `estimate_loss`
averages 200 random batches (`train.py:216`), reproducing four decimals is not the bar.
Landing within about 0.01 is.

(The numerically stable form actually used in kernels is `logsumexp(z) − z_t`, and the full
treatment of perplexity, bits-per-byte, and the unit conversions between them is
`loss-and-optimization.md`. Here we only need the derivative.)

### 3.2 The backward pass is a tape, and the tape is your memory bill

Autograd is neither symbolic differentiation nor finite differences. During the forward
pass PyTorch records, for every operation, a node holding (a) a function that computes the
vector-Jacobian product and (b) references to the input tensors that function will need.
That recording is the **tape** (a Wengert list). `loss.backward()` walks it in reverse
topological order, and each node's contribution is accumulated into `.grad` on the leaves.

Chain rule, written once with the symbols named:

```
∂ℓ/∂x  =  (∂y/∂x)ᵀ · ∂ℓ/∂y
```

- `x` — the input to some operation
- `y` — its output
- `∂ℓ/∂y` — the gradient already computed for the output ("upstream gradient")
- `(∂y/∂x)ᵀ` — the transposed Jacobian of that operation; never materialized, only applied

The systems consequence is the whole reason activations dominate the memory budget: the
node for `y = W x` needs `x` to compute `∂ℓ/∂W = ∂ℓ/∂y · xᵀ`. So **`x` must stay resident
from the moment it is produced in the forward pass until the corresponding backward node
runs.** For the first layer of a 48-layer model, that is the entire forward and almost the
entire backward pass. Activation memory is not a transient; it is a working set whose
lifetime is the full step, and its size is set by batch × sequence, not by parameter count.

Activation checkpointing `[C]` (arXiv 1604.06174, Apr 2016) is the standard trade: do not
save `x`, save only the block boundary, and recompute the block's interior during backward.
You pay roughly one extra forward pass of FLOPs to buy back most of the activation memory.
The systems reflex is "spill the cold data to a slower tier" — and that is exactly what this
is *not*. Nothing is written anywhere. The data is destroyed and reconstructed from a seed
state. Which means the price is denominated in compute, not bandwidth, and — a detail that
matters on unvalidated hardware — the recomputed activation is not guaranteed bit-identical
to the original, because the recompute may pick a different kernel.

### 3.3 AdamW, symbol by symbol

Per parameter `θ`, at step `t`, with gradient `g_t`:

```
m_t = β₁ · m_{t−1} + (1 − β₁) · g_t                  first moment  (an EMA of the gradient)
v_t = β₂ · v_{t−1} + (1 − β₂) · g_t²                 second moment (an EMA of the squared gradient)

m̂_t = m_t / (1 − β₁^t)                               bias correction
v̂_t = v_t / (1 − β₂^t)

θ_t = θ_{t−1} − η · ( m̂_t / (√v̂_t + ε)  +  λ · θ_{t−1} )
```

- `g_t` — this step's gradient for this parameter
- `m_t` — exponentially weighted average of past gradients; `β₁ = 0.9` means a ~10-step window
- `v_t` — exponentially weighted average of past *squared* gradients; `β₂ = 0.95` in LLM
  practice (not torch's 0.999 default) means a ~20-step window, because LLM gradients are
  noisy and a long window tracks badly
- `m̂`, `v̂` — bias correction. `m_0 = v_0 = 0`, so early estimates are biased toward zero;
  dividing by `1 − β^t` undoes exactly that. At `t = 1` with `β₁ = 0.9` the divisor is 0.1,
  i.e. a 10× correction. This is why step 1 is not tiny.
- `η` — learning rate
- `ε` — a small constant (default 1e-8) preventing division by zero. `[C]` arXiv 2509.02046
  (Sep 2025) shows published optimizer rankings *flip* when `ε` is tuned rather than
  defaulted; treat it as a hyperparameter, not a guard.
- `λ` — weight decay coefficient

**The division is the whole idea.** `m̂ / √v̂` is dimensionless: if you scaled every gradient
for a parameter by 1000, both numerator and denominator scale by 1000 and the step is
unchanged. So the update size is set by `η` and by gradient *consistency*, not gradient
*magnitude*. A parameter with a tiny but steady gradient moves as far as one with a huge
noisy gradient. That is the property SGD lacks and the reason Adam works on transformers,
where gradient magnitudes vary by orders of magnitude across layers.

**What the "W" fixed.** Adam originally applied weight decay by adding `λθ` to the gradient
*before* the moment updates — so the decay term went through the `/√v̂` division and got
scaled per-parameter by an amount that has nothing to do with regularization. AdamW applies
`λθ` directly to the weight, outside the division. That is the entire difference, and it is
why the term sits outside the fraction above.

**Which parameters get decay** is a config decision with a real convention behind it.
nanoGPT's rule is purely dimensional: `p.dim() >= 2` decays, everything else does not
(`training/nanogpt/model.py:270`). That decays LayerNorm weights? No — those are 1-D, so they
are excluded, correctly. But it *does* decay the embedding table, which is 2-D. The frontier
convention (`research/notes/pretraining-recipes.md` §1) is decay on matmul matrices but not on
norms, biases, **or embeddings**. So nanoGPT diverges from the convention, silently, via a
dimensionality test. For shakespeare_char the split is 10,740,096 decayed / 4,992 not.

### 3.4 Where every byte lives — worked, for a real config

Take nanoGPT's `train_shakespeare_char.py` exactly as shipped
(`training/nanogpt/config/train_shakespeare_char.py:22`): `n_layer=6`, `n_head=6`,
`n_embd=384` (call it `d`), `block_size=256` (`T`), `batch_size=64` (`B`), `vocab=65` (`V`),
`bias=False`, weights tied.

**Parameter count**, computed rather than quoted:

| Tensor | Shape | Count |
|---|---|---|
| `wte` (tied with `lm_head`) | 65 × 384 | 24,960 |
| `wpe` | 256 × 384 | 98,304 |
| per block: `ln_1`, `ln_2` | 2 × 384 | 768 |
| per block: `c_attn` | 384 × 1152 | 442,368 |
| per block: `c_proj` (attn) | 384 × 384 | 147,456 |
| per block: `c_fc` | 384 × 1536 | 589,824 |
| per block: `c_proj` (mlp) | 1536 × 384 | 589,824 |
| **per block total** | | **1,770,240** |
| × 6 blocks | | 10,621,440 |
| `ln_f` | 384 | 384 |
| **total `P`** | | **10,745,088** |

nanoGPT prints 10.65M because `get_num_params` subtracts `wpe` by default
(`model.py:150`). Both numbers are right; they answer different questions. Use the full
10,745,088 for memory and the non-embedding 10,646,784 for FLOPs.

**Persistent state, at 4 bytes each (nanoGPT keeps parameters in fp32 and uses autocast):**

| What | Bytes | MiB |
|---|---|---|
| parameters (fp32) | 4P = 42,980,352 | 40.99 |
| gradients (fp32) | 4P = 42,980,352 | 40.99 |
| Adam `m` (fp32) | 4P | 40.99 |
| Adam `v` (fp32) | 4P | 40.99 |
| **total = 16 bytes/param** | **171,921,408** | **163.96** |

Sixteen bytes per parameter is the number to memorize. Note that
`research/notes/pretraining-recipes.md` §1 also arrives at 16 B/param but with a *different
composition* — bf16 weights (2) + bf16 grads (2) + fp32 master (4) + `m` (4) + `v` (4) —
which is the FSDP mixed-precision path. Same total, different tensors, and therefore a
different transient peak. Under `torch.autocast` there is no persistent bf16 weight copy;
instead each weight is cast to bf16 on first use inside the autocast region and the cast is
cached until the region exits, adding up to `2P` bytes of *transient* footprint (21 MiB here).

**Activations.** Assume for a moment that `scaled_dot_product_attention` takes a fused path —
the FlashAttention construction `[C]` (arXiv 2205.14135, May 2022), which tiles the softmax so
the `T × T` score matrix is never written to memory and is recomputed during backward from the
saved log-sum-exp. Then the tensors the tape must keep per
token per layer are: the residual entering `ln_1` (`d`), `ln_1`'s output (`d`), the fused QKV
(`3d`), the attention output (`d`), the residual entering `ln_2` (`d`), `ln_2`'s output (`d`),
the MLP pre-activation (`4d`), and the GELU output (`4d`). That is ≈ **16·d elements per token
per layer**.

`[M]` Measured, and the count holds: instrumenting the tape with
`torch.autograd.graph.saved_tensors_hooks` (the method Exercise A uses) on the CPU config with
`dropout=0` gives **16.7·d elements per token per layer** — the extra 0.7·d is LayerNorm's
saved mean/rstd, SDPA's log-sum-exp, and the log-softmax the loss keeps. Single run, CPU,
`torch 2.11.0+cu128`, 2026-07-26; an anecdote by the house standard, and structural rather
than performance-sensitive, which is why it is quoted at all.

Converting elements to bytes is where the naive answer goes wrong. Autocast does **not** put
everything in bf16 — LayerNorm, softmax, and cross-entropy stay in fp32 — so the saved set is
mixed. `[M]` The same run measured **11.2 MB bf16 + 3.85 MB fp32**, an average of
**2.29 bytes per element**, i.e. ≈ **38·d bytes per token per layer**, not the 32·d you get by
assuming bf16 throughout. Scaled to the GPU config (`d = 384`, `L = 6`, 16,384 tokens):

```
38 · d bytes/token/layer  ≈  38 × 384         = 14,592 B/token/layer
× L = 6                                       = 87,552 B/token
× (B·T) = 64 × 256 = 16,384 tokens            ≈ 1.34 GiB
```

**And now the finding that actually matters, which contradicts the paragraph above.**
`[M]` Same instrument, CPU config (`d=128`, `L=4`, `H=4`, `B=12`), sweeping `T` and reporting
**bytes per token** so the sequence-length term is visible rather than hidden in the total:

| `dropout` | T=32 | T=64 | T=128 | T=256 |
|---|---|---|---|---|
| 0.0 | 21,716 | 19,640 | 18,603 | 18,084 B/token |
| 0.2 | 35,476 | 39,544 | 50,795 | **74,852 B/token** |

At `dropout=0` the per-token cost is flat in `T` — activation memory is `O(T)` overall, which
is what a fused attention kernel buys you. **At `dropout=0.2` the per-token cost grows roughly
linearly in `T`, which means total activation memory is `O(T²)`.** Nonzero attention dropout
pushes `scaled_dot_product_attention` off the fused path onto the math path, which materializes
the full `B × H × T × T` score matrix and its dropout mask and saves both for backward.
`training/nanogpt/config/train_shakespeare_char.py` ships **`dropout = 0.2`**, so the reference
config quietly defeats flash attention. Measured on CPU, `torch 2.11.0+cu128`, single run.

**On our actual GPU it is worse, and this is already measured.** `[M]` `tensors-and-autograd.md`
§2.4 fits `saved(T) = k + a·T + c·T²` on gfx1151 (`B=16`, fp32, 2026-07-26, single run) and
finds that **`scaled_dot_product_attention` dispatches to the math backend by default** —
its quadratic coefficient is 2,304 B/token², bit-for-bit the same as nanoGPT's hand-written
attention path, matching the prediction for exactly one saved `(B, nh, T, T)` fp32 tensor per
layer. PyTorch reports `flash_sdp_enabled() == True` and `mem_efficient_sdp_enabled() == True`
and materialises the score matrix anyway. Setting
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` drops the coefficient to **exactly 0.0**.

So on this machine the `O(T²)` term is the default regardless of dropout, and the 1.34 GiB
linear estimate above is a floor, not a total. Scale the quadratic term yourself:
`c = L · B · n_head · bytes`, so for the GPU config (`L=6`, `B=64`, `n_head=6`, fp32 scores)
`c = 9,216 B/token²` — **576 MiB at `T=256`, and 36 GiB at `T=2048`**, which is past the
`[M]` 32 GiB single-tensor fault and would present as a hang.

**What is still open, and it is the sharp version of the question.** The AOTriton measurement
was taken at `dropout=0`, and the CPU measurement above says that nonzero attention dropout is
precisely what defeats a fused path. Nobody has checked whether AOTriton's memory-efficient
kernel *keeps* `c = 0` when `dropout_p > 0` — and the shipped reference config sets
`dropout = 0.2`. If it does not, then setting the environment variable buys nothing for the
recipe we actually run, and every long-context arm has to zero attention dropout as well.
That is Exercise A, part three.

**The logits tensor, which is the one that will actually kill you.** Cross-entropy runs in
fp32 under autocast, so the logits are `B·T·V·4` bytes, and its gradient is the same size
again:

| Vocab | logits bytes | + gradient |
|---|---|---|
| 65 (shakespeare_char) | 4.06 MiB | 8.1 MiB |
| 50,304 (GPT-2) | 3.07 GiB | 6.14 GiB |

At the *same* 16,384 tokens, a GPT-2 vocabulary makes the loss head 5.5× larger than every
activation in the transformer combined. **The loss head is sized by vocabulary, not by
parameters**, and at ablation scale it is the largest single tensor in the step. This is
exactly the failure mode flagged in `research/notes/pretraining-recipes.md` §9: with a
100k vocab and a 131,072-token microbatch it reaches 49 GiB in one tensor, and
`ASSUMPTIONS.md: large-tensor-fault-32gib` records `[M]` that a 32 GiB single tensor
**hangs at 0% CPU with no error** on this machine. The naive fix — raise the microbatch to
use the big memory — is precisely what triggers it.

**What scales with what.** This is the capacity-planning summary:

| Term | Scales with | On the Z13 |
|---|---|---|
| params + grads + optimizer state | `P` only | 4.8 GB at 300M — 7% of the `[M]` ≥62 GiB fast tier |
| activations | `B · T · L · d` — **but `B · T² · L · H` if attention dropout is on** | the only term you can trade against batch size |
| logits + logit grad | `B · T · V` | the tensor that hits the `[M]` 32 GiB fault first |
| tape metadata | number of ops | negligible |

`[M]` fast tier ≥62 GiB at ~200 GB/s, `notebook/uma-carveout-controls-fast-tier.md`,
single run per arm. `[M]` 32 GiB single-tensor hang, same date. Both in `ASSUMPTIONS.md`.

### 3.5 Gradient accumulation, and the normalization that is usually wrong

You want a global batch of `N` loss-bearing tokens but only `N/G` fit in memory. So you run
`G` micro-batches, each doing its own forward and backward, letting the gradients pile up in
`.grad`, and step once at the end. `backward()` accumulates rather than assigns — that is
not a convenience, it is the mechanism.

OLMo-core implements this as a **spatial split**, not a loop counter: `split_batch` chops the
rank's batch by `rank_microbatch_size` (`train_module.py:393`) and the number of accumulation
steps is derived. That framing is the correct one, and it makes the next problem visible.

The loss you want is the mean over all loss-bearing tokens in the *global* batch:

```
L = (1/N) · Σ_{n=1}^{N} ℓ_n
```

- `N` — total number of tokens in the global batch that contribute to the loss (i.e. not
  padding, not `ignore_index`)
- `ℓ_n` — per-token cross-entropy

The common implementation computes a per-micro-batch mean and divides by `G`:

```
L_naive = (1/G) · Σ_{g=1}^{G} [ (1/n_g) · Σ_{n ∈ g} ℓ_n ]
```

- `n_g` — number of loss-bearing tokens in micro-batch `g`

**When every `n_g` is equal**, `n_g = N/G`, and:

```
L_naive = (1/G) · Σ_g (G/N) · Σ_{n∈g} ℓ_n = (1/N) · Σ_n ℓ_n = L      ✓
```

**When they are not equal, it is wrong**, because it weights each micro-batch equally
regardless of how many tokens it contains. Worked, with numbers:

Two micro-batches: 1000 loss tokens averaging 2.0, and 100 loss tokens averaging 4.0.

```
correct:   (1000·2.0 + 100·4.0) / 1100  =  2400/1100  =  2.1818
naive:     (2.0 + 4.0) / 2              =  3.0
```

37.5% too high — and since the gradient is linear in the loss, the gradient is wrong in
exactly the same way. Per token, the naive scheme applies weight `1/(G·n_g)`:

```
short micro-batch token:  1/(2·100)  = 1/200   vs correct 1/1100  →  5.5× over-weighted
long  micro-batch token:  1/(2·1000) = 1/2000  vs correct 1/1100  →  0.55× under-weighted
```

A 10× relative distortion — exactly `n_long / n_short`. This is not hypothetical. `[C]`
It was found in October 2024 across most popular LLM trainers, reported by Benjamin Marie,
publicized by Unsloth on 2024-10-15, and patched in HuggingFace Transformers the next day
(github.com/huggingface/transformers PR #34191). `[C]` The same defect persisted in RL
trainers well after: verl fixed it in November 2025; secondary reports as of mid-2026 say
OpenRLHF and Llama-Factory had not.

**The fix**, and it is worth reading in the reference code because it is not what you would
guess. OLMo-core does not divide by `G` at the end. It divides by the *whole batch's* token
count **inside** each micro-batch's loss, using a sum reduction:

```
loss_reduction = "sum"                              # train_module.py:406
loss_div_factor = batch_num_tokens_for_loss         # train_module.py:408
```

so each micro-batch contributes `(Σ_{n∈g} ℓ_n) / N`, and the accumulated total is exactly
`L`. The usual "divide by accumulation steps" step does not exist in that codebase at all.
This is correct for ragged micro-batches by construction, and it is a genuinely better
design than the standard formulation.

nanoGPT uses the naive form (`train.py:301`, `loss = loss / gradient_accumulation_steps`).
It is *correct there* — every micro-batch has exactly `batch_size × block_size` loss tokens,
because `get_batch` slices fixed-length windows with no padding and no masking. It is
correct for a reason the code never states, which is the most dangerous kind of correct.

**Where the write-combining analogy breaks.** Deferring a cache-line writeback and deferring
a gradient all-reduce look identical (`train_module.py:566` suppresses cross-rank reduction
on every micro-batch but the last). The difference is what happens when you get the flush
point wrong: a missed writeback eventually surfaces as a stale read; a missed final sync
silently trains each rank on its own local gradient and the run diverges with no error at
all. Irrelevant to us — `ASSUMPTIONS.md: single-device-only` — but the shape of the failure
is the lesson.

### 3.6 Mixed precision: the bit layouts, and the two arguments

Three formats, three integers each:

| Format | Sign | Exponent | Mantissa | Max finite | Min normal | Unit roundoff `u = 2^−(m+1)` |
|---|---|---|---|---|---|---|
| fp32 | 1 | 8 | 23 | ≈3.40e38 | ≈1.18e−38 | 2⁻²⁴ ≈ 5.96e−8 |
| fp16 | 1 | 5 | 10 | 65,504 | 2⁻¹⁴ ≈ 6.10e−5 | 2⁻¹¹ ≈ 4.88e−4 |
| bf16 | 1 | 8 | 7 | ≈3.39e38 | ≈1.18e−38 | 2⁻⁸ ≈ 3.91e−3 |

- **mantissa bits** set *relative* precision: how finely you can resolve a value once you
  know its scale. `u` is the largest relative error a single rounding can introduce.
- **exponent bits** set *dynamic range*: how large and how small a value can be at all.

bf16 is fp32 with 16 mantissa bits deleted. It has fp32's range and 1/65536 of fp32's
relative resolution. fp16 spends 3 of its bits on mantissa instead of exponent, so it is
**8× more precise than bf16 and catastrophically narrower in range**.

**Argument one, why fp16 needs a loss scaler.** LLM gradients routinely have magnitudes
below fp16's smallest normal, `2⁻¹⁴ ≈ 6.1e−5`, and a meaningful tail below its smallest
subnormal, `2⁻²⁴ ≈ 6.0e−8`, where they flush to exactly zero. A zeroed gradient is not a
small error — it is a parameter that stops learning. The fix is loss scaling: multiply the
loss by `S` before `backward()`. By linearity of differentiation every gradient is then
scaled by `S`, moving the whole distribution up into representable range; divide by `S`
before the optimizer step. That is precisely nanoGPT's three lines:

```
scaler.scale(loss).backward()      # train.py:305   multiply by S, then backward
scaler.unscale_(optimizer)         # train.py:308   divide by S before clipping
scaler.step(optimizer)             # train.py:311   step, unless any grad was inf/nan
```

Dynamic loss scaling is a control loop: on any inf/nan, discard the step and halve `S`; after
2000 clean steps, double it. Bridge: it is a congestion-control loop with multiplicative
increase and multiplicative decrease. **Where it breaks:** TCP's loss signal is a symptom of
congestion elsewhere and costs you latency; the scaler's overflow signal is a hard local
failure and costs you an entire optimizer step of work that is thrown away. And the scaler is
a *stateful* component you must checkpoint, or a resume silently restarts the search for `S`.

**Argument two, why bf16 needs fp32 master weights.** bf16's range means gradient underflow
is not the failure mode, so `scaler` is a no-op (`train.py:196`: `enabled=(dtype == 'float16')`).
What bf16 pays is precision, and it pays it at exactly the worst place — the weight update.

Concretely. Near `w = 1.0`, adjacent bf16 values differ by `2⁻⁷ = 0.0078125`. Round-to-nearest
means any increment smaller than half that, `2⁻⁸ = 0.00390625`, leaves `w` **exactly
unchanged**. A typical update `η·ĝ` with `η = 1e−3` and a normalized Adam step of order 1 is
about `1e−3`... but late in training, with per-parameter steps of order `1e−5`:

```
threshold / update  =  0.00390625 / 1e−5  =  390.6
```

The update is 390× below the rounding threshold. It does not partially apply. It vanishes,
and it vanishes on *every* step, forever. This is "swamping," and it is the entire reason
mixed-precision training keeps an fp32 copy of the weights `[C]` (arXiv 1710.03740, Oct 2017,
the paper that introduced the fp32-master-weights + loss-scaling recipe): **the accumulator
must be more precise than the operand.** The bf16 copy is what the matmuls read; the fp32
copy is what the optimizer writes.

There is an alternative that removes the fp32 copy: **stochastic rounding** — round up with
probability proportional to how far you are through the interval, so an update 1/390 of a
ulp moves the weight with probability 1/390 and accumulates correctly *in expectation*.
`[C]` arXiv 2502.20566 (Feb 2025) reports BF16+SR beating (BF16, FP32) mixed precision up to
6.7B params on perplexity, throughput, and memory, and being more robust to high learning
rates. `[A]` Low confidence that this is usable here: it needs either hardware SR support or
a custom optimizer kernel, and neither is validated on gfx1151. Exercise C part 1 emulates
it in five lines so you can see the mechanism.

**Where bf16 bites even with master weights** — the honest list, all of which are open on
this machine:

1. `[C]` **RoPE at large position indices.** bf16's 7-bit mantissa cannot represent large
   position values accurately, so RoPE's relative-position property degrades as context
   grows and the error accumulates during training (arXiv 2411.13476, Nov 2024). Compute
   position math in fp32 and cast after.
2. `[C]` **Flash-attention-style kernels specifically.** arXiv 2510.04212 (Oct 2025,
   ICLR 2026) traces low-precision training failure to biased rounding-error accumulation
   in the attention kernel.
3. `[C]` **Late-training loss spikes may be a rounding artifact.** arXiv 2605.06152
   (May 2026) argues the periodic "slingshot" spikes come from logit differences exceeding
   the absorption-error threshold, so correct-class gradients round to zero while
   incorrect-class ones do not, breaking gradient balance. New, unreplicated, and directly
   about the derivative in §3.1.
4. `[M]` **Our own hardware is unproven.** `ASSUMPTIONS.md: bf16-numerics-unproven` is still
   `untested`, against `[C]` five documented bf16 bugs on gfx1151 (ROCm #6034). No result
   from this machine counts as evidence until the Hardware Validation Gate runs.

**And the contested part, which is new.** The "bf16 for training, always" consensus has one
serious 2025–2026 crack: `[C]` arXiv 2510.26788 (Oct 2025) argues that for RL fine-tuning,
bf16's rounding error is what creates the training/inference policy mismatch, and that
simply reverting to fp16 eliminates it — validated across GRPO/GSPO/TIS/MIS/PG, several model
families, and two independent frameworks. That is a post-training claim, not a pretraining
one, and it does not overturn the pretraining default. But it is the first credible argument
in years that the format choice is a decision rather than a given, and if this lab ever runs
a post-training arm it is directly load-bearing.

### 3.7 Determinism: what a seed buys, and what it cannot

Sources of run-to-run variation, in the order you should eliminate them:

1. **RNG.** Weight init, dropout masks, and data sampling all draw from PRNGs. Python's
   `random`, NumPy's, and torch's CPU and CUDA generators are *separate streams* and all
   must be seeded — that is exactly what `seed_all` does
   (`training/olmo-core/src/olmo_core/utils.py:164`). nanoGPT seeds only torch
   (`train.py:106`), which is sufficient only because it uses no other RNG.
2. **Float non-associativity.** `(a + b) + c ≠ a + (b + c)` in floating point. A GPU
   reduction splits the sum across threads and combines partial results in whatever order
   they complete, so the same inputs give different bits depending on the launch geometry.
   This is not a bug and cannot be seeded away.
3. **Atomics.** Several backward kernels (embedding backward, `scatter_add`, `index_add`)
   use `atomicAdd`, whose completion order is genuinely nondeterministic. Same input, same
   launch config, different bits.
4. **Library autotuning.** hipBLASLt/cuBLAS pick kernels heuristically; Triton autotune
   caches results per shape. The chosen kernel can differ across processes.
5. **Batch invariance.** `[C]` Thinking Machines Lab, "Defeating Nondeterminism in LLM
   Inference" (Sep 2025) — the same row of a matmul produces different output depending on
   how many *other* rows are in the batch, because the reduction split changes. They built
   batch-invariant RMSNorm/matmul/attention kernels and got 1000 identical runs out of 1000.
   The result is about inference; **no equivalent exists for the training loop.**
   `[C]` arXiv 2506.09501 (Jun 2025, rev. Oct 2025) quantifies the same effect and reports
   bf16 as markedly worse than fp16 or fp32 for reproducibility, with accuracy swings of
   several points from tensor-parallel size and batch size alone.

**The bridge and its break.** Idempotent replay in a distributed system works because you can
make operations commutative or impose an order. Neither is available here: float addition is
*not* associative, and the hardware reorders reductions specifically to go fast. Your options
are to pin the reduction order (`torch.use_deterministic_algorithms(True)`) and pay in
throughput and coverage gaps, or to accept bitwise drift.

**Which brings the distinction that actually matters for this lab.** There are two different
properties and they are routinely conflated:

- **Bitwise reproducibility** — same seed, same bits. Useful for exactly two things:
  regression tests, and checkpoint round-trip verification. `CLAUDE.md`'s Hardware Validation
  Gate requires the checkpoint round-trip to be **bit-exact**, and CODE_MAP explains why it
  must be checked on *weights* rather than on loss trajectories: nanoGPT's resume restores
  model, optimizer, `iter_num` and `best_val_loss` and nothing else
  (`training/nanogpt/train.py:179`) — never the RNG state, never a data position, because
  `get_batch` samples offsets with replacement from the global RNG (`train.py:116`). A
  resumed run therefore diverges from an uninterrupted one *by construction*.
- **Seed-controlled comparability** — two arms differ only by the arm under test, and
  seed-to-seed variance is measured rather than assumed. **This is what makes an ablation
  valid, and bitwise determinism is neither necessary nor sufficient for it.** The house rule
  of ≥3 seeds with confidence intervals is the actual control. A bitwise-deterministic
  single-seed comparison is still an anecdote.

Separate the two seeds in the config. Data order and weight init should be independently
controllable, because "does this result survive a different data order?" and "does it
survive a different initialization?" are different questions.

### 3.8 The FLOPs arithmetic, and fixing nanoGPT's MFU

One forward pass through a weight matrix costs 2 FLOPs per parameter per token (one multiply,
one add). Backward costs about twice that — one matmul for the gradient with respect to the
input, one for the gradient with respect to the weight. Hence the universal estimate:

```
C ≈ 6 · N · D
```

- `C` — total training FLOPs
- `N` — number of (non-embedding) parameters
- `D` — number of training tokens

nanoGPT adds the attention term, which `6N` misses because attention's cost depends on
sequence length rather than on parameters (`training/nanogpt/model.py:296`):

```
flops_per_token = 6·N + 12·L·H·Q·T
```

- `L` — layers, `H` — heads, `Q` — head dimension (so `H·Q = d`), `T` — sequence length
- the `12·L·d·T` term is `QKᵀ` and `AV`: `4·T·d` FLOPs per token per layer forward, ×3 for
  forward+backward

For shakespeare_char (`N = 10,646,784`, `L = 6`, `d = 384`, `T = 256`):

```
6N              = 63,880,704
12·L·d·T        = 12 × 6 × 384 × 256 = 7,077,888
flops_per_token = 70,958,592                     attention share = 9.97%
per iteration   = 70,958,592 × 256 × 64 = 1.163e12 FLOP
```

Attention is 10% of the cost at `T = 256`. At `T = 2048` on the same model it is 47%. That
is the quadratic term becoming the whole story, and it is the entire economic motivation for
the SWA/global hybrids in the memory track.

(The derivation of the `6` — two facts about matmul and one about backprop — and what the rule
leaves out is `scaling-laws-and-flops-budget.md`. Here it is a budgeting tool.)

**Now the trap.** nanoGPT divides achieved FLOP/s by a hardcoded `flops_promised = 312e12`
(`model.py:301`) — A100 bf16 peak. On the 8060S that denominator is meaningless and the
printed MFU will read absurdly low even when the stack is configured correctly. `[M]` Our
measured bf16 GEMM ceiling is **20.9 TFLOP/s at 8192³** (`scripts/benchmark_gemm.py`,
2026-07-26, `ASSUMPTIONS.md: gemm-throughput-below-reference`), so the correction factor is
`312 / 20.9 = 14.93×`. Patch the denominator before using that number as a health signal.

---

## 4. Why it matters for Proteus

**The config surface is the experimental surface, and the loop contributes more fields to it
than people expect.** Every knob below is a real ablation axis *and* a real confound if it
silently differs between arms. Matched budgets means matched loop config, not just matched
params and tokens.

| Config field | Why it is on the surface |
|---|---|
| `precision.autocast_dtype` | bf16 default. fp16 becomes a live option if we ever run a post-training arm `[C]` (2510.26788). |
| `precision.master_weights_dtype` | fp32, per §3.6. Making it a field is what lets us *test* the swamping claim rather than assume it. |
| `precision.fp32_reductions` | Loss reduction, softmax accumulation, and **RoPE position math** in fp32 `[C]` (2411.13476). Three separate booleans, because we will want to attribute which one mattered. |
| `batch.global_batch_tokens` / `batch.rank_microbatch_tokens` | Accumulation steps are *derived*, following OLMo-core's spatial-split design (`train_module.py:393`), never set directly. |
| `loss.div_factor_policy` | `global_token_count` is the default and the only correct one for ragged batches. `per_microbatch_mean` exists solely so a pre-registered arm can *reproduce* the 2024 bug and measure its size at our scale. |
| `loss.chunk_size` | Assert `T_micro × V × 4 ≤ 8 GiB` in the config validator. `[M]` 32 GiB single tensors hang at 0% CPU with no error. |
| `optim.zero_grad_set_to_none` | Not a pure optimization — see below. |
| `determinism.init_seed`, `determinism.data_order_seed` | Two independent seeds. Different questions. |
| `telemetry.metrics_collect_interval` | Observability costs throughput here; the interval is a real tradeoff, not a logging preference. |
| `activation_checkpointing.granularity` | The recompute-vs-store price is different on unified memory than on HBM. See §8. |

**Three Proteus-specific consequences worth calling out.**

**One: `set_to_none` is a semantic switch on MoE arms.** PyTorch's own documentation for
`Optimizer.zero_grad` states it plainly `[C]`: *"torch.optim optimizers have a different
behavior if the gradient is 0 or None (in one case it does the step with a gradient of 0 and
in the other it skips the step altogether)."* Adam's implementation gates on
`if p.grad is not None`. For a dense model this never matters — every parameter gets a
gradient every step. For `proteus-moe-sigmoid`, an expert that receives no tokens in a
micro-batch has either a zero gradient or no gradient depending on how the layer is written
and on this flag, and the difference is real: with `None`, the expert is skipped entirely —
no weight decay, no momentum decay. With zeros, AdamW *does* step it: `λθ` shrinks it and
`m` continues to push it. `[A]` Medium confidence that this materially changes expert-collapse
dynamics over a full run. **Cheapest test:** a 20M MoE arm, two runs differing only in this
flag, tracking per-expert weight norm and routing entropy. Hours, not days.

**Two: the loss head, not the model, is the memory hog at our scale, and Laguna's vocabulary
is large.** Everything in §3.4's logits table applies directly to any Proteus arm with a real
tokenizer. This is also the term that interacts with the capacity story: the Z13's `[M]`
≥62 GiB fast tier invites large microbatches, and the largest microbatch is bounded by
`T_micro × V × 4` hitting `[M]` 32 GiB, not by anything about the model.

**Three: the loop is where the `[A]` 6 TFLOP/s assumption gets closed.**
`research/notes/pretraining-recipes.md` §5 builds its entire wall-clock and rent-versus-run
table on an assumed sustained throughput of 6 TFLOP/s, and names it as the note's riskiest
assumption. Every exercise below runs the loop; instrumenting tokens/s while you are in there
is nearly free and converts that `[A]` into `[M]`.

---

## 5. Read the code

Two codebases: nanoGPT because it is the whole loop in one readable file, and OLMo-core
because it is what the same loop looks like when it has to survive a real run. Read them in
that order. Paths are relative to `research/reference/`; run `scripts/fetch_reference.sh`
first if the clones are not materialized.

### nanoGPT — the loop with nothing hidden

| Where | What to look at, and why |
|---|---|
| `training/nanogpt/train.py:255` | `while True:` — the outer loop. Read to `:333` in one sitting. This is the complete training loop of a real LLM trainer, and it fits on a screen. Notice there is no framework, no callbacks, and no abstraction between you and the six statements. |
| `training/nanogpt/train.py:292` | The micro-batch loop. Compare to OLMo-core's spatial split below: here accumulation genuinely *is* a loop counter, which is why line 301 has to divide by it. |
| `training/nanogpt/train.py:301` | `loss = loss / gradient_accumulation_steps` — the mean-of-means normalization from §3.5. Ask yourself why it is correct here, then check `get_batch` at `:116` to confirm your answer (fixed-length slices, no padding, no `ignore_index`). |
| `training/nanogpt/train.py:305` | `scaler.scale(loss).backward()` — the only place the loss scaler touches the loss, and the only place the backward pass is invoked. |
| `training/nanogpt/train.py:314` | `optimizer.zero_grad(set_to_none=True)` — placed *after* the step, with a comment explaining it is done to free memory. This is the accumulation boundary; §4's `set_to_none` discussion is about this exact line. |
| `training/nanogpt/train.py:196` | `GradScaler(enabled=(dtype == 'float16'))` — the whole bf16-vs-fp16 story in one predicate. bf16 disables the scaler because it does not need it. Note this API spells `torch.cuda.amp.GradScaler`, deprecated since torch 2.4 in favour of `torch.amp.GradScaler('cuda', ...)`; you will see a FutureWarning on our 2.12 wheel. |
| `training/nanogpt/train.py:73` | The dtype auto-selection: bf16 if the device supports it, else fp16. Read it as the field's decision recorded in one line. |
| `training/nanogpt/train.py:112` | The autocast context. Everything inside runs in mixed precision; everything outside (the optimizer step) does not. That boundary is the design. |
| `training/nanogpt/train.py:106` | `torch.manual_seed(1337 + seed_offset)` — the *only* seeding in the file. Compare to `seed_all` below and list what is missing. |
| `training/nanogpt/train.py:323` | `lossf = loss.item()` with the comment "this is a CPU-GPU sync point". One `.item()` inside the hot loop. Hold this next to OLMo-core's `record_metric`. |
| `training/nanogpt/train.py:179` | The resume path restores `iter_num` — and, reading upward, model/optimizer/`best_val_loss` and nothing else. No RNG state, no data cursor. This is why the checkpoint gate must compare weights bit-exactly rather than loss curves. |
| `training/nanogpt/model.py:187` | `F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-1)` — §3.1 in one line, including the reshape that flattens batch and time into one axis. |
| `training/nanogpt/model.py:263` | `configure_optimizers` — the param-group split. |
| `training/nanogpt/model.py:270` | `decay_params = [... if p.dim() >= 2]` — the dimensional rule that decays the embedding table, diverging from the frontier convention. A three-word predicate carrying an unstated design decision. |
| `training/nanogpt/model.py:296` | `flops_per_token = 6*N + 12*L*H*Q*T` — §3.8's arithmetic, as shipped. |
| `training/nanogpt/model.py:301` | `flops_promised = 312e12` — the hardcoded A100 denominator. Change it to `20.9e12` `[M]` before trusting the printed MFU. |
| `training/nanogpt/model.py:56` | `q, k, v = self.c_attn(x).split(...)` — K and V computed fresh every forward and discarded. **There is no KV cache in training.** |
| `training/nanogpt/README.md:51` / `:85` | The two published targets: 1.4697 on GPU, 1.88 on the CPU config. These are the Hardware Validation Gate's pass/fail numbers. |

### OLMo-core — the same loop, hardened

| Where | What to look at, and why |
|---|---|
| `training/olmo-core/src/olmo_core/train/trainer.py:1466` | `_fit_epoch` — the control plane. It advances `global_step`, fires callbacks, and calls `train_batch → optim_step → zero_grads`. It never touches a tensor. Compare the shape to nanoGPT's `while True` and note what the separation buys. |
| `training/olmo-core/src/olmo_core/train/trainer.py:1474` and `:1506` | `zero_grads()` appears twice: once before the loop and once after every `optim_step`. The pre-loop call is the one people forget; it makes the first step correct after a resume. |
| `training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:393` | `split_batch` — accumulation as a *spatial split* by `rank_microbatch_size`, derived from a memory budget. Not a counter. This is the better design and it is why the next two lines can exist. |
| `training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:406` and `:408` | `loss_reduction="sum"` and `loss_div_factor=batch_num_tokens_for_loss`. **This is the fix from §3.5.** Read these two lines, then go back to nanoGPT `train.py:301` and articulate the difference out loud. |
| `training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:422` | `loss.backward()` inside the micro-batch loop — one backward per micro-batch, accumulating into `.grad`. |
| `training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:596` | `torch.autocast(self.device.type, dtype=self.autocast_precision)` — mixed precision as a config field rather than a global. Contrast nanoGPT's module-level `ctx`. |
| `training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:514` | `optim_step` — grad-norm clipping, LR schedule application per param group, `optim.step()`, and `SkipStepOptimizer` handling for loss/grad-norm spikes. Everything nanoGPT does inline, named and ordered. |
| `training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:538` | `zero_grads` — one line, `set_to_none=True`. |
| `training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:566` | `_train_microbatch_context` — cross-rank reduction suppressed on all but the last micro-batch. Irrelevant on one device, but it is the write-combining pattern and the comment explains the flush point. |
| `training/olmo-core/src/olmo_core/utils.py:164` | `seed_all` — seeds `random`, `numpy`, `torch`, and `torch.cuda` explicitly, with a comment saying the last call is deliberately redundant. Diff this against nanoGPT's single `manual_seed` and you have the list of streams nanoGPT leaves loose. |
| `training/olmo-core/src/olmo_core/train/trainer.py:1517` | `torch.cuda.set_sync_debug_mode("warn")` — after every metrics flush, arm a tripwire so any *accidental* host-device sync in the training path gets reported. |
| `training/olmo-core/src/olmo_core/train/utils.py:137` | `with cuda_sync_debug_mode(0):` in `move_metrics` — the one place the tripwire is disarmed, because that sync is intentional. Read these two pointers as a pair: this is what "observability costs throughput" looks like when someone has actually engineered around it. |
| `training/olmo-core/src/olmo_core/train/trainer.py:1037` / `:1394` | `record_metric` buffers metrics as *unevaluated device tensors*; `_log_metrics` drains them in one host-device sync and hands reduction to a background thread. Group commit with none of a WAL's durability — the buffer is volatile and the batching exists to avoid a stall, not to amortize I/O. |
| `training/olmo-core/src/olmo_core/data/data_loader.py:667` | `_build_global_indices` — data order as a pure function of `(seed, epoch, dataset length)`, never persisted. The dataloader stores a cursor, not a log. This is the determinism property nanoGPT lacks. |

---

## 6. Exercises

All three run on the Z13 (gfx1151, native Windows, one GPU). Activate with
`. .\scripts\activate-lab.ps1` from the repo root — it puts `C:\venvs\lab` on PATH and sets
`HIPBLASLT_TENSILE_LIBPATH` / `TORCH_BLAS_PREFER_HIPBLASLT`. Work from
`research/reference/training/nanogpt/` and run `python data/shakespeare_char/prepare.py` once.

> **`[M]` Read this before you activate.** `curriculum/tokenization.md` measured, 2026-07-26,
> reproduced twice: with `TORCH_BLAS_PREFER_HIPBLASLT=1` and `HIPBLASLT_TENSILE_LIBPATH` set —
> exactly what `activate-lab.ps1` exports — a bf16 matmul of shape `[T,768] @ [768,V]`
> **segfaults inside `libhipblaslt.dll`** at `hipblasLtMatmulAlgoGetHeuristic()` (0xC0000005).
> Clearing both variables makes the identical shapes run clean. The square 8192³ GEMM in
> `scripts/benchmark_gemm.py` does *not* crash, so it is shape-dependent. **The shape it hits
> is the language-model head**, which every exercise below runs. If a run dies with an access
> violation and no Python traceback, clear both variables in the session and retry before
> assuming your code is wrong — and record which setting you used, because
> `ASSUMPTIONS.md: hipblaslt-config` says the variables buy only +12% on this wheel.

**ROCm/Windows caveats that apply to all three:**
- Pass `--compile=False`. `torch.compile` on gfx1151/Windows is not validated here and a
  compile failure will masquerade as a model bug.
- **Set `$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"` unless you are deliberately
  measuring without it.** `[M]` Without it, `scaled_dot_product_attention` on gfx1151 silently
  takes the math backend and materializes the full score matrix, even though
  `flash_sdp_enabled()` reports `True` (`tensors-and-autograd.md` §2.4). It enables a backend
  AMD labels experimental, so state in any run record whether it was set.
- You will see a `FutureWarning` for `torch.cuda.amp.GradScaler`. Harmless.
- `train.py:107` sets `torch.backends.cuda.matmul.allow_tf32 = True`. `[A]` Medium
  confidence this is inert on gfx1151 — RDNA has no TF32/xf32 datapath. Cheapest test: time
  an fp32 matmul with the flag on and off.
- `ASSUMPTIONS.md: large-tensor-fault-32gib` — keep every single tensor under 32 GiB. A hang
  here presents at 0% CPU with no error, so a runaway batch size looks like a stalled script.
- **CPU fallback for all three:** substitute the published CPU config
  (`training/nanogpt/README.md:85`) — `n_layer=4 n_head=4 n_embd=128 block_size=64
  batch_size=12`. Every measurement below is device-independent by construction; the numbers
  change, the method does not.

### Exercise A — Account for every byte in one step

**Difficulty 2/5. ~45–75 min, mostly writing the hooks; the run itself is seconds.**

Predict, then measure, the four persistent state tensors and the activation working set.
Use `torch.autograd.graph.saved_tensors_hooks` to intercept every tensor the tape saves —
this works identically on CPU and GPU, which is why it is the right instrument rather than
`torch.cuda.max_memory_allocated()` alone.

```python
# scratch/state_inventory.py  -- run from research/reference/training/nanogpt
import torch
from model import GPT, GPTConfig

DEV = "cuda" if torch.cuda.is_available() else "cpu"
cfg = GPTConfig(vocab_size=65, block_size=256, n_layer=6, n_head=6,
                n_embd=384, dropout=0.0, bias=False)   # CPU: block 64, 4 layers, 4 heads, 128
model = GPT(cfg).to(DEV)                               # dropout is the variable in part three
opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.99), DEV)

def mib(ts):                       # dedup by storage: views alias, don't double count
    seen, tot = set(), 0
    for t in ts:
        if t is None: continue
        p = t.untyped_storage().data_ptr()
        if p in seen: continue
        seen.add(p); tot += t.numel() * t.element_size()
    return tot / 2**20

B, T = 64, 256                                                 # CPU: 12, 64
x = torch.randint(0, 65, (B, T), device=DEV)
y = torch.randint(0, 65, (B, T), device=DEV)

saved = []
with torch.autograd.graph.saved_tensors_hooks(lambda t: (saved.append(t), t)[1],
                                              lambda t: t), \
     torch.autocast(DEV, dtype=torch.bfloat16):
    logits, loss = model(x, y)

print(f"params        {mib(model.parameters()):8.2f} MiB")
print(f"saved-for-bwd {mib(saved):8.2f} MiB")
loss.backward()
print(f"grads         {mib([p.grad for p in model.parameters()]):8.2f} MiB")
opt.step()
st = [v for s in opt.state.values() for v in s.values() if torch.is_tensor(v)]
print(f"optim state   {mib(st):8.2f} MiB")
if DEV == "cuda":
    print(f"peak alloc    {torch.cuda.max_memory_allocated()/2**20:8.2f} MiB")
```

**What to check, part one — the fixed state.** Params, grads, and optimizer state should read
**40.99 / 40.99 / 81.98 MiB** — that is `4P`, `4P`, `8P` for `P = 10,745,088`, totalling the
163.96 MiB of §3.4. These are exact, not approximate. If they do not match, your parameter
count is wrong and you should find that out before anything else. While you are here, print
`loss.item()` on the untouched model and confirm it lands near `ln 65 = 4.174`.

**What to check, part two — the batch sweep.** With `dropout=0.0`, loop
`B ∈ {8, 16, 32, 64, 128}` and plot saved-for-backward and peak allocation against `B`. Both
should be straight lines through approximately the origin, with a fixed ~164 MiB offset for
peak. Extract the slope in bytes/token. §3.4 predicts ≈ **87,552 B/token** for this config —
`[A]` medium confidence, since that is the `[M]` CPU-measured 38·d scaled up, and GPU autocast
keeps a different set of ops in fp32. **That slope is the number you will use for every
microbatch-sizing decision in this lab, so get it from your own machine rather than from this
document.**

**What to check, part three — the genuinely open one.** Sweep `T ∈ {64, 128, 256}` and fit
`saved(T) = k + a·T + c·T²`, exactly as `tensors-and-autograd.md` §2.4 does. Run the fit in
**four** conditions: `{AOTriton off, on} × {dropout 0.0, 0.2}`, where "on" means
`$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"` set before launching Python.

What is already known `[M]` (`tensors-and-autograd.md`, gfx1151, 2026-07-26, `dropout=0`):
AOTriton off gives `c = 2,304 B/token²`; AOTriton on gives `c = 0.0`. Your job is the other
two cells. The prediction, stated so it can fail: **`c` returns to nonzero at `dropout=0.2`
even with AOTriton on**, because attention dropout requires materializing a mask over the
score matrix and that is exactly what defeated the fused path on CPU (§3.4).

Either answer is worth having and neither is in any document this lab has found:

- If `c` stays 0 under dropout, the environment variable alone fixes the reference recipe and
  the only cost is running an AMD-experimental backend.
- If `c` goes nonzero, then **`dropout` and `attention memory complexity` are coupled config
  fields on this hardware**, every long-context arm must set attention dropout to 0 and say
  so, and the `[M]` 32 GiB single-tensor fault becomes reachable at surprisingly modest `T`.

Write the 2×2 table into `notebook/` either way. It changes the microbatch ceiling for every
long-context experiment in the memory track.

**Bonus, nearly free:** time 50 steps, compute `tokens/s`, multiply by
`flops_per_token = 70,958,592` from §3.8, and report TFLOP/s. That is the direct attack on
`research/notes/pretraining-recipes.md` open question #1 — the `[A]` 6 TFLOP/s sustained
throughput assumption the whole cost model rests on.

### Exercise B — Break gradient accumulation, then fix it

**Difficulty 3/5. ~30–45 min. Reproduces a bug that shipped in most LLM trainers for years.**

Build one reference gradient from a single full batch, then reproduce it two ways from
micro-batches: with the mean-of-means normalization and with the global-token-count
normalization. Do it first with uniform micro-batches, then with ragged ones.

```python
# scratch/accumulation_normalization.py
import torch, torch.nn.functional as F
from model import GPT, GPTConfig

torch.manual_seed(1337)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
cfg = GPTConfig(vocab_size=65, block_size=64, n_layer=2, n_head=2,
                n_embd=64, dropout=0.0, bias=False)     # dropout 0 => no RNG divergence
model = GPT(cfg).to(DEV)

def flat_grad(m):
    return torch.cat([p.grad.reshape(-1) for p in m.parameters() if p.grad is not None])

def run(mbs, mode):
    model.zero_grad(set_to_none=True)
    N = sum(int((y != -1).sum()) for _, y in mbs)          # global loss-token count
    for x, y in mbs:
        logits, _ = model(x, y)
        fl, fy = logits.view(-1, logits.size(-1)), y.view(-1)
        if mode == "mean_of_means":
            loss = F.cross_entropy(fl, fy, ignore_index=-1) / len(mbs)
        else:
            loss = F.cross_entropy(fl, fy, ignore_index=-1, reduction="sum") / N
        loss.backward()
    return flat_grad(model).clone()

def make(n_seq, n_masked):     # mask the first n_masked positions of each sequence
    x = torch.randint(0, 65, (n_seq, 64), device=DEV)
    y = torch.randint(0, 65, (n_seq, 64), device=DEV)
    y[:, :n_masked] = -1
    return x, y

def compare(a, b, label):
    cos = F.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).abs().max() / b.abs().max()).item()
    print(f"{label:24s} cos={cos:.10f}  max_rel_err={rel:.3e}")

uniform = [make(8, 0) for _ in range(4)]                  # 4 x 512 loss tokens
ragged  = [make(8, 0), make(8, 0), make(8, 0), make(8, 56)]  # 512,512,512,64
for name, mbs in (("uniform", uniform), ("ragged", ragged)):
    ref = run([ (torch.cat([x for x,_ in mbs]), torch.cat([y for _,y in mbs])) ], "global")
    compare(run(mbs, "global"),        ref, f"{name}: global      ")
    compare(run(mbs, "mean_of_means"), ref, f"{name}: mean_of_means")
```

**What to check.** `[M]` A CPU dry-run of exactly this script (`torch 2.11.0+cu128`, seed
1337, single run, 2026-07-26) produced:

| Case | cosine vs. full-batch reference | max relative error |
|---|---|---|
| uniform `[512,512,512,512]`, `global` | 1.0000082 | 3.0e−7 |
| uniform `[512,512,512,512]`, `mean_of_means` | 1.0000082 | 3.0e−7 |
| ragged `[512,512,512,64]`, `global` | 1.0000085 | 3.6e−7 |
| ragged `[512,512,512,64]`, `mean_of_means` | **0.6705** | **1.65** |

Four things to take from that table.

1. **Uniform, both modes agree** with the reference to ~7 decimals. Both normalizations are
   correct when micro-batches are equal-sized, exactly as §3.5 proves.
2. **The agreement is not bitwise, and cosine similarity comes out slightly *above* 1.0.**
   That is not a bug in the script; it is float error in the cosine itself, and it is the
   batch-invariance problem from §3.7 showing up in your own output. The full-batch reference
   and the four-micro-batch version reduce over different shapes and therefore in different
   orders.
3. **Ragged with the `global` divisor is still correct** — 1.0000085, unchanged. The
   formulation is correct by construction for unequal micro-batches.
4. **Ragged with `mean_of_means` has a cosine of 0.67 and a relative error of 165%.** That is
   not a rounding artifact; the gradient is pointing in a materially different direction. With
   loss-token counts `[512, 512, 512, 64]`, §3.5 predicts the short micro-batch's tokens carry
   `2048/256 = 8×` their correct weight. Reproduce it and note whether your numbers agree.

One number — 0.67 — that says "your trainer was silently computing a different gradient."

Then go read `training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:406`–`:408` again. The fix is two
keyword arguments.

### Exercise C — Precision and repeatability on this specific machine

**Difficulty 3/5. ~60–90 min. Feeds two open rows in `ASSUMPTIONS.md`.**

**Part 1 — measure the bf16 swamping threshold, then defeat it.** No model needed; runs
anywhere in seconds.

```python
import torch
torch.manual_seed(0)
upd = 1e-5

w_bf, w_f32 = torch.tensor([1.0], dtype=torch.bfloat16), torch.tensor([1.0])
for _ in range(100_000):
    w_bf = (w_bf.float() + upd).to(torch.bfloat16)   # round-to-nearest, into bf16
    w_f32 += upd                                     # accumulate in fp32
print("bf16 accumulator :", w_bf.item())             # 1.0
print("fp32 accumulator :", w_f32.item())            # ~2.0014

def sr_to_bf16(x_f32):        # bf16 IS the top 16 bits of an fp32, so SR is three lines:
    bits  = x_f32.contiguous().view(torch.int32)     # add uniform noise to the 16 bits
    noise = torch.randint(0, 1 << 16, x_f32.shape, dtype=torch.int32)   # about to be dropped,
    return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)  # then truncate

w_sr = torch.tensor([1.0], dtype=torch.bfloat16)
for _ in range(100_000):
    w_sr = sr_to_bf16(w_sr.float() + upd)
print("bf16 + stochastic rounding :", w_sr.item())   # ~1.99
```

**What to check.** `[M]` Verified on CPU with `torch 2.11.0+cu128`, 2026-07-26, single run:
the bf16 accumulator reads **exactly 1.0** after 100,000 additions — one hundred thousand
updates, zero movement — the fp32 accumulator reads **2.0013580** (the residual is because
`1e−5` is not exactly representable), and the stochastic-rounding version reaches
**1.9921875**, which is exactly one bf16 ulp below 2.0. Verify the threshold arithmetic
yourself: `torch.finfo(torch.bfloat16).eps` is `2⁻⁷ = 0.0078125`, half of that is
`0.00390625`, and `0.00390625 / 1e−5 = 390.6`.

That `1.0` is the entire argument for fp32 master weights, and that `1.9921875` is the entire
argument for `[C]` arXiv 2502.20566 — the *same* 7-bit format recovers the right answer once
you stop rounding to nearest. Note also the trap in the naive implementation: writing the SR
interval with `torch.nextafter` gives you the next **fp32**, not the next **bf16**, so `p`
comes out ~65,536× too small and the whole thing silently degrades back to round-to-nearest,
printing `1.0` and looking like it works. Bit-twiddling on the fp32 representation is the
correct formulation because bf16 is literally a truncated fp32.

**Part 2 — where do our gradients actually sit relative to fp16's floor?** Train the
shakespeare_char config for 500 steps, then histogram gradient magnitudes at step 0 and step
500 against the two fp16 thresholds:

```python
g = torch.cat([p.grad.abs().reshape(-1).float() for p in model.parameters()
               if p.grad is not None])
for name, thr in [("< fp16 min subnormal 2^-24", 2**-24), ("< fp16 min normal 2^-14", 2**-14)]:
    print(f"{name}: {(g < thr).float().mean().item():.4%}")
print("max |g| =", g.max().item(), " (fp16 max finite = 65504)")
```

**What to check.** Report both fractions at both steps. The claim under test — stated as a
prediction so it can fail — is that the fraction below `2⁻¹⁴` *rises* between step 0 and step
500, because the output gradient shrinks as `p_t → 1` (§3.1). If it rises, you have measured
why fp16 needs a loss scaler and bf16 does not, on your own hardware. If it does not, that is
a finding about small-vocabulary character models and worth writing down.

**Part 3 — determinism, and the honest version of it.** Three sub-measurements, each one line
of conclusion:

1. **Run-to-run.** Two fresh processes, same seed, 20 steps, no dropout. Compare final
   parameters with `torch.equal`. Report bit-identical yes/no, and if no, the max absolute
   difference. Do it once in fp32 and once under bf16 autocast, and compare the two answers —
   `[C]` arXiv 2506.09501 predicts bf16 is markedly worse.
2. **Determinism coverage on ROCm.** Set `torch.use_deterministic_algorithms(True)` and rerun.
   **Record every op that raises `RuntimeError` for having no deterministic implementation.**
   That list is a direct `[M]` contribution to the Hardware Validation Gate's determinism
   item, and as far as this lab knows it has never been written down for gfx1151.
3. **Checkpoint round-trip.** Save, reload into a fresh model, and assert every parameter is
   bit-identical. Per §3.7 and CODE_MAP, this must compare *weights*, not loss trajectories —
   nanoGPT restores neither RNG state nor data position (`train.py:179`), so a resumed run
   diverges from an uninterrupted one by construction and a loss-based check would fail for
   the wrong reason.

Anything you measure here that bears on bf16 belongs in `ASSUMPTIONS.md` under
`bf16-numerics-unproven`, tagged `[M]` with the wheel version
(`torch 2.12.0a0+rocm7.13.0a20260313`, HIP 7.2.0, driver 32.0.23033.5002) — and remember the
house standard: one run per arm is an anecdote and must be labelled as one.

---

## 7. Self-check

1. You delete the `optimizer.zero_grad()` line. Nothing raises. Describe the mechanism in
   terms of what `.grad` actually is, and give the effective learning rate at step `k`.

2. Write down the bit layout of bf16 and fp16. Explain, using those numbers, why fp16 needs a
   loss scaler and bf16 does not, and what bf16 pays instead. Give the threshold below which a
   weight update near 1.0 rounds away entirely.

3. Your micro-batches contain `[1000, 100]` loss-bearing tokens with mean losses `[2.0, 4.0]`.
   Compute the correct batch loss and the mean-of-means loss. By what factor is a token in the
   short micro-batch over-weighted in the gradient?

4. Two runs with the same seed on the same machine agree to 6 decimal places at step 10 and to
   3 by step 500. Is the rig broken? What would you have to change to make them bit-identical,
   and should you?

5. In the memory budget for one training step, which terms scale with batch size and which do
   not? On the Z13 specifically, which term hits a hard limit first, at what number, and why is
   the obvious mitigation the thing that triggers it?

6. `zero_grad(set_to_none=True)` versus `False` makes no behavioural difference for a dense
   model. Name the Proteus arm where it does, and state exactly what changes.

*(Answers at the end of this file.)*

---

## 8. What is still unsolved here

The loop looks settled. Five things in it are not.

**The gradient-accumulation normalization is mathematically settled and still broken in
shipping code.** §3.5 is not a research question — it is arithmetic. Yet `[C]` the defect was
industry-wide until October 2024, verl fixed its RL variant only in November 2025, and
secondary reports as of mid-2026 say OpenRLHF and Llama-Factory still carry it. The open
problem is not the math; it is that *"correct in the literature" and "correct in the code you
are about to run" are different states*, and nobody publishes which frameworks are currently
in which state. For a lab that reads source, this is cheap to check and expensive to assume.

**bf16 is the default by consensus, not by controlled comparison at our scale.** The consensus
formed when the alternative was fp16 plus a loss scaler and the models were huge. Three live
cracks: `[C]` arXiv 2510.26788 (Oct 2025) argues fp16 is strictly better for RL post-training
because bf16's rounding error *is* the training/inference mismatch; `[C]` arXiv 2502.20566
(Feb 2025) argues BF16+stochastic-rounding beats the (BF16, FP32) recipe outright up to 6.7B;
`[C]` arXiv 2605.06152 (May 2026) argues that periodic late-training loss spikes are a
finite-precision artifact rather than an optimization phenomenon. None of the three has been
replicated at 20M–300M, and all three are directly testable here. Do not present the format
choice as closed.

**Determinism in the *training* loop has no batch-invariant answer.** `[C]` Thinking Machines
(Sep 2025) solved batch-invariance for inference with custom RMSNorm/matmul/attention kernels
and got 1000-for-1000 identical outputs. There is no published equivalent for the backward
pass, where atomics in embedding and scatter backward are an additional source, and nobody has
published what bitwise-deterministic training costs in throughput at any scale. `[C]` arXiv
2506.09501 makes it worse by showing bf16 is the *most* variance-prone of the three formats —
so the format the field standardized on for training is also the one that reproduces least
well. Whether that matters for ablation validity (as opposed to debugging) is, as far as this
survey found, unaddressed.

**Nothing in the literature prices activation checkpointing on unified memory.** Every
recompute-versus-store analysis assumes HBM is scarce and DRAM is far away. On the Z13 they
are the same pool: `[M]` ≥62 GiB of fast tier at ~200 GB/s with no PCIe hop
(`notebook/uma-carveout-controls-fast-tier.md`). That inverts the usual tradeoff — storing an
activation may be cheaper than recomputing it well past the point where the literature says to
switch — and it interacts with `[M]` the 32 GiB single-tensor fault, which caps how large any
single stored activation may be regardless of pool size. This is a genuinely open,
lab-specific question with a cheap experiment attached: sweep checkpointing granularity at
fixed batch and plot step time against peak memory.

**And a smaller one that is entirely ours to close.** Two `[M]` measurements meet here and
leave a gap. `tensors-and-autograd.md` §2.4 shows that on gfx1151 the backend-availability flags do not
describe what actually runs: `scaled_dot_product_attention` reports
`flash_sdp_enabled() == True` and takes the math backend anyway, giving `O(T²)` activation
memory until `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` drives the quadratic coefficient to
exactly zero. §3.4 here shows that on CPU, nonzero **attention dropout** is what defeats a
fused path. Nobody has measured the intersection — whether AOTriton keeps `c = 0` under
`dropout_p > 0` — and the shipped reference config sets `dropout = 0.2`. It is a twenty-minute
measurement (Exercise A, part three) that decides the microbatch ceiling for every long-context
experiment in the memory track. It is also a good illustration of the general shape of this
whole section: published activation-memory formulas are written for a kernel-selection regime
nobody states, and on unvalidated hardware you have to measure which regime you are in.

**Whether any of this registers above seed noise at 20M–300M is our own open assumption.**
`ASSUMPTIONS.md: ablation-scale-sufficient` is `untested`, `[A]` medium confidence, and
`research/memory/open-problems-ranked.md` names it the riskiest assumption in the backlog. It
applies with full force here: if the loop-level effects above are smaller than the
seed-to-seed spread at our scale, they are unmeasurable in this lab and belong on the rented-
hardware list. The `[A]` 6 TFLOP/s sustained-throughput figure in
`research/notes/pretraining-recipes.md` §5 is the gate on finding out, because it decides how
many seeded arms fit in the wall-clock budget — which is why Exercise A asks you to record
tokens/s while you are already in the loop.

---

## Answers to the self-check

**1.** `.grad` is a persistent per-parameter accumulator, and `backward()` *adds into* it
(`grad += new_grad`) rather than assigning. It is not a return value; it is a side effect on
a buffer that outlives the call. Without a reset, at step `k` the buffer holds the sum of `k`
gradients. AdamW normalizes by `√v̂`, which is also inflated, so the effect is not a clean
`k×` on the step — but the *gradient fed to the optimizer* is `k` times too large and grows
without bound, and in the SGD case the effective learning rate is exactly `k·η`. It presents
as a run that trains normally for a few steps and then diverges, which is indistinguishable
from a learning rate that is too high. No exception is ever raised. This is the accumulation
mechanism working exactly as designed — the design just requires you to declare the boundary
(`train.py:314`, `train_module.py:538`).

**2.** bf16 = 1 sign + 8 exponent + 7 mantissa; fp16 = 1 sign + 5 exponent + 10 mantissa.
Exponent bits set dynamic range, mantissa bits set relative precision. fp16's 5 exponent bits
give a smallest normal of `2⁻¹⁴ ≈ 6.1e−5` and a largest finite value of 65,504; LLM gradients
routinely fall below that floor and flush to zero, so the loss is multiplied by `S` before
backward (every gradient scales by `S` by linearity) and divided by `S` before the step. bf16
has fp32's 8 exponent bits, so underflow is not the failure mode and the scaler is disabled
(`train.py:196`). What bf16 pays is 3 fewer mantissa bits — unit roundoff `2⁻⁸ ≈ 3.9e−3`
versus fp16's `2⁻¹¹ ≈ 4.9e−4`, so 8× coarser. Near `w = 1.0` the bf16 spacing is
`2⁻⁷ = 0.0078125`, so under round-to-nearest **any update smaller than `2⁻⁸ = 0.00390625`
leaves the weight bit-identical**. That is why the optimizer's accumulator is kept in fp32.

**3.** Correct: `(1000×2.0 + 100×4.0) / 1100 = 2400/1100 = 2.1818`. Mean-of-means:
`(2.0 + 4.0)/2 = 3.0`, which is 37.5% high. Per-token weights: naive gives `1/(G·n_g)`, so
`1/200` for the short micro-batch against a correct `1/1100` → **5.5× over-weighted**; the
long micro-batch's tokens get `1/2000` against `1/1100` → 0.55×. The *relative* distortion
between the two groups is 10×, which is exactly `n_long/n_short`. The fix is to divide by the
global loss-token count inside each micro-batch and use a sum reduction
(`train_module.py:406`, `:408`).

**4.** Not broken — expected. The seed fixes every PRNG stream but cannot fix float
non-associativity: GPU reductions combine partial sums in completion order, atomics in
embedding/scatter backward complete nondeterministically, and library autotuning may pick
different kernels per process. Divergence in the last bits at step 10 amplifies through 490
more nonlinear steps; 3 decimals at step 500 is normal, not pathological. To force
bit-identity you would need `torch.use_deterministic_algorithms(True)`, deterministic
reduction kernels, fixed launch geometry, and disabled autotuning — costing throughput and,
on ROCm, hitting ops with no deterministic implementation at all (Exercise C part 3 measures
which). **You should not**, in general. Bitwise determinism buys you two things: regression
tests and the bit-exact checkpoint round-trip the Hardware Validation Gate requires. It does
*not* make an ablation valid. What makes an ablation valid is ≥3 seeds with a reported
confidence interval, so that arm-to-arm differences are compared against measured seed-to-seed
spread. A bitwise-deterministic single-seed comparison is still an anecdote.

**5.** Fixed in batch size: parameters, gradients, and optimizer moments — `16 bytes × P`
total, 164 MiB for nanoGPT's 10.7M and 4.8 GB at 300M, which is ~7% of the `[M]` ≥62 GiB fast
tier. Scaling with batch: activations, `[M]` measured at ≈`38·d` bytes per token per layer
(87,552 B/token for this config) — **plus a `B·T²·L·n_head` term that is live by default on
gfx1151**, because SDPA takes the math backend unless
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` is set — and the logits plus their gradient, at
`B·T·V·4` bytes each. **The logits tensor hits the wall first at our sequence lengths**,
because it is sized by *vocabulary* rather than by model size: at 16,384 tokens and a
50,304-entry vocab it is 3.07 GiB, and `[M]` a single tensor of 32 GiB hangs at 0% CPU with no
error on this machine (`ASSUMPTIONS.md: large-tensor-fault-32gib`). The obvious mitigation for
having 62 GiB of fast memory is to raise the microbatch — which is precisely what drives
`T_micro × V × 4` into the fault. Chunk the cross-entropy over the sequence axis and assert
`T_micro × V × 4 ≤ 8 GiB` in the config validator. **And know where the crossover is:** logits
plus gradient cost `8·B·T·V` bytes, the saved score matrices cost `4·L·B·n_head·T²`, so they
are equal at `T = 2V / (L · n_head)`. For this 6-layer, 6-head config with a GPT-2 vocab that
is `T ≈ 2,800`; for a 24-layer, 16-head model it is `T ≈ 262`. Deeper and wider models cross
over far earlier, which is why the AOTriton environment variable is not optional for any
long-context arm.

**6.** `proteus-moe-sigmoid` (or any MoE arm). PyTorch's optimizers gate on
`if p.grad is not None`, and the `Optimizer.zero_grad` docstring states the consequence
explicitly `[C]`: with a zero gradient the optimizer takes the step, with `None` it skips
entirely. An expert that receives no tokens in a micro-batch therefore behaves differently:
with `set_to_none=True` it is skipped — **no decoupled weight decay, no momentum decay, no
`v` decay** — while with `set_to_none=False` AdamW steps it anyway, shrinking it by `λθ` and
continuing to apply whatever momentum `m` still carries. Over a full run that is a systematic
difference in how unrouted experts drift, which is exactly the mechanism expert-collapse
studies are trying to measure. `[A]` Medium confidence that it changes the outcome materially;
cheapest test is two 20M MoE runs differing only in the flag, tracking per-expert weight norm
and routing entropy.

---

## Sources

`[C]` arXiv ids verified against the live arXiv API on 2026-07-26; resolution proves the paper
exists, not that it supports the claim beside it.

**Optimization and the loop.** arXiv 1412.6980 — Adam: A Method for Stochastic Optimization
(Dec 2014). arXiv 1711.05101 — Decoupled Weight Decay Regularization (Nov 2017). arXiv
2509.02046 — Fantastic Pretraining Optimizers and Where to Find Them (Sep 2025; the `ε`-tuning
result). arXiv 1812.06162 — An Empirical Model of Large-Batch Training (Dec 2018).

**Memory and the tape.** arXiv 1604.06174 — Training Deep Nets with Sublinear Memory Cost
(Apr 2016). arXiv 2205.14135 — FlashAttention: Fast and Memory-Efficient Exact Attention with
IO-Awareness (May 2022).

**Precision.** arXiv 1710.03740 — Mixed Precision Training (Oct 2017; fp32 master weights +
loss scaling). arXiv 2010.06192 — Revisiting BFloat16 Training (Oct 2020). arXiv 2502.20566 —
Stochastic Rounding for LLM Training: Theory and Practice (Feb 2025). arXiv 2510.26788 —
Defeating the Training-Inference Mismatch via FP16 (Oct 2025). arXiv 2510.04212 — Why
Low-Precision Transformer Training Fails: An Analysis on Flash Attention (Oct 2025, ICLR 2026).
arXiv 2605.06152 — Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes
(May 2026). arXiv 2411.13476 — When Precision Meets Position: BFloat16 Breaks Down RoPE in
Long-Context Training (Nov 2024). arXiv 2405.18710 — To FP8 and Back Again: Quantifying Reduced
Precision Effects on LLM Training Stability (May 2024).

**Determinism.** arXiv 2506.09501 — Understanding and Mitigating Numerical Sources of
Nondeterminism in LLM Inference (Jun 2025, rev. Oct 2025). Thinking Machines Lab, "Defeating
Nondeterminism in LLM Inference" (thinkingmachines.ai/blog, 11 Sep 2025) — industry blog post
with released code, not a peer-reviewed paper; cited as such.

**Stability.** arXiv 2309.14322 — Small-scale proxies for large-scale Transformer training
instabilities (Sep 2023).

**Non-arXiv.** The gradient-accumulation normalization defect: reported by Benjamin Marie,
publicized by Unsloth (unsloth.ai/blog/gradient, 15 Oct 2024), patched in
github.com/huggingface/transformers PR #34191 (16 Oct 2024); persistence in RL trainers
(verl fixed Nov 2025) from secondary reporting surveyed 2026-07-26 and not confirmed against
each project's source. PyTorch `Optimizer.zero_grad` docstring, on the zero-versus-None
optimizer behaviour.

**Local.** `research/reference/CODE_MAP.md` (nanoGPT gate recipe; OLMo-core trainer, train
module, dataloader). `ASSUMPTIONS.md` — `gpu-fast-tier-size` (`[M]` ≥62 GiB at ~200 GB/s),
`large-tensor-fault-32gib` (`[M]` 32 GiB hang), `gemm-throughput-below-reference` (`[M]` 20.9
TFLOP/s bf16 at 8192³), `bf16-numerics-unproven` (untested), `single-device-only`,
`ablation-scale-sufficient`. `ENVIRONMENT.md` (version pins).
`notebook/uma-carveout-controls-fast-tier.md`. `research/notes/pretraining-recipes.md` (§1
optimizer state arithmetic, §5 the wall-clock table and the `[A]` 6 TFLOP/s assumption, §9 the
logits-tensor cliff and the telemetry schema). `research/memory/kv-cache-mechanics.md` (the
`[M]` ~105 FLOP/byte ridge point, and the training-versus-decode regime distinction).
`research/memory/open-problems-ranked.md`.

---

## Decision / Riskiest assumption / Next test

**Decision.** Adopt the loop as specified: bf16 autocast with fp32 master weights, fp32 for
loss reduction, softmax accumulation, and RoPE position math; gradient accumulation as a
spatial split with `loss_reduction="sum"` and a global-token-count divisor; `set_to_none=True`
with the MoE caveat recorded; independent init and data-order seeds; a config validator that
asserts `T_micro × V × 4 ≤ 8 GiB`.

**Riskiest assumption.** That bf16 is numerically trustworthy on gfx1151 at our shapes
(`ASSUMPTIONS.md: bf16-numerics-unproven`, still `untested`, against `[C]` five documented
bf16 bugs). Every number this module tells you to measure is measured *in* bf16.

**Next test.** Exercise C, in full, on the Z13 — the swamping threshold, the gradient-magnitude
histogram, and the determinism triple including the list of ops that have no deterministic
ROCm implementation. It is the cheapest work in this module and it is the only work here that
converts an open `ASSUMPTIONS.md` row.
