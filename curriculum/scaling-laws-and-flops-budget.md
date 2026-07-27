---
title: Scaling laws and the FLOPs budget — deriving 6ND, budgeting a run, and reading MFU honestly
version: 1.0.0
date: 2026-07-26
track: A — Foundations
owner: curriculum-author
prereqs: transformer-forward-pass-by-hand, the-training-loop
difficulty: medium (the arithmetic is easy; the conventions are where people get hurt)
time: ~4 hours reading and working the math, plus 2–5 hours of exercises
---

# Scaling laws and the FLOPs budget

## What this module settles

**One:** the 6·N·D rule is not a heuristic — it falls out of two facts about matrix
multiplication and one about backpropagation, and once you have derived it you can also
say exactly what it leaves out and when the omission matters. **Two:** Chinchilla's "20
tokens per parameter" is one specific answer from one specific fit that its own authors'
third estimation method disagrees with, and the 2026 literature has moved from "find a
better law" to "the fitting procedure is biased" — which lands directly on a lab that
runs narrow, small sweeps. **Three:** you can now cost any run on this machine before
starting it, and the one number that decides every scheduling and rent-versus-run
decision in the lab is currently `[A]`, not `[M]` — this module tells you exactly which
number that is and what it would take to measure it.

---

## Theory in plain language

You are about to start a batch job. You want to know two things before you start it: how
long it will take, and whether the output will be any good. In every system you have
operated, those questions are answered by different people with different tools —
capacity planning tells you the runtime, and a spec tells you whether the output is
correct. Training breaks that separation. The size of the job *is* the quality of the
output, continuously and without a threshold, and the function relating them was
discovered empirically less than six years ago and is still argued about.

The cost side turns out to be almost embarrassingly simple. A transformer's forward pass
is a chain of matrix multiplications. Everything else in it — layer norms, softmax,
activation functions, residual adds — costs a rounding error by comparison. And a matrix
multiply has a rigid FLOP count you can read off its shapes. So the total arithmetic in a
training run is a closed-form function of "how many weights" times "how many tokens",
with a small correction term. That is `C ≈ 6ND`. It is the only cost model in ML that a
capacity planner would recognise as legitimate: no fudge factor, no benchmark, just
counting.

The quality side is where it gets uncomfortable. Given a fixed compute budget `C`, you
can spend it on a big model trained on few tokens, or a small model trained on many. The
loss you end up with depends on the split. Kaplan et al. in 2020 `[C]` (arXiv 2001.08361,
Jan 2020) fitted power laws and concluded you should mostly spend on parameters. Hoffmann
et al. in 2022 `[C]` (arXiv 2203.15556, Mar 2022 — "Chinchilla") redid it more carefully
and concluded parameters and tokens should scale roughly together, at around 20 tokens
per parameter, and demonstrated it by training a 70B model on 1.4T tokens that beat a
280B model. Most of the Kaplan/Chinchilla discrepancy has since been attributed to
Kaplan counting non-embedding parameters where Chinchilla counted all of them `[C]`
(arXiv 2406.12907, Jun 2024) — a bookkeeping choice, not a disagreement about physics.
Hold onto that: **the definition of `N` is load-bearing, and it bit the two most cited
papers in the field.**

What replaced Chinchilla? Nothing, cleanly. The 2026 position is that the law's *fitting
methodology* is biased `[C]` (arXiv 2603.22339, Mar 2026), that the original numbers did
not fully replicate `[C]` (arXiv 2404.10102, Apr 2024), and that the objective was wrong
anyway if you intend to serve the model, because inference cost pushes you toward smaller
models trained far past compute-optimal `[C]` (arXiv 2401.00448, Dec 2023; arXiv
2501.18107, Jan 2025; arXiv 2604.01411, Apr 2026). The field fragmented into conditional
laws — per data quality, per optimizer, per MoE sparsity — rather than converging on a
successor. This is consistent with `research/notes/pretraining-recipes.md`, which reaches
the same conclusion from the recipe side.

### The bridge, and where it breaks

The bridge for `6ND` is a **cost model for a batch job**: rows × bytes × passes, the
thing you write on a whiteboard before provisioning an ETL. It is a good bridge. It
breaks in one place: an ETL has a fixed amount of work determined by the input, and you
size the cluster to fit it. Here *you choose the amount of work*, and the amount you
choose determines what the artifact is. There is no "correct" C. There is only a budget
and a tradeoff curve you cannot see until you have paid.

The bridge for **MFU** is sharper and it is the one to internalise, because MFU is the
metric you will be reading on every dashboard for the next year:

> **MFU is "logical bytes written ÷ the datasheet's sequential-write throughput."**

The numerator is not work the machine did. It is work the machine *should have had to
do*, computed analytically from the model's shapes — the logical write. The denominator
is a vendor peak that no real workload reaches — the datasheet number. Between them sits
everything a storage engineer would call write amplification: recomputed activations,
padding, redundant compute in tensor parallelism, kernel launch gaps. The metric that
counts the *physical* FLOPs actually issued is called HFU (hardware FLOPs utilization),
and HFU ≥ MFU always, with the gap being implementation overhead `[C]` (the accounting is
worked in arXiv 2205.05198, May 2022; MFU itself is defined in the PaLM paper, arXiv
2204.02311, Apr 2022, which is the source olmo-core's implementation cites).

**Where the analogy breaks, and it is the interesting part:** in storage, write
amplification is pure waste you would eliminate if the flash translation layer let you.
In training, the amplification is often a *deliberate purchase*. Activation
recomputation burns extra FLOPs to buy memory, and buying that memory can let you run a
larger microbatch that raises end-to-end tokens/second. So a *lower* MFU/HFU ratio can
accompany a *faster* run. Nothing in an observability stack behaves that way. A second
break: in ops, high utilization is usually a warning; here it is the goal. A third: the
denominator is a marketing number, and vendors quote the sparsity-enabled figure.
Olmo-core's speed monitor literally multiplies NVIDIA's published H100 numbers by
`dense_correction = 0.5` before using them
(`training/olmo-core/src/olmo_core/train/callbacks/speed_monitor.py:99`). Two labs
reporting "45% MFU" may not be reporting the same quantity.

---

## The math that actually matters

### One matmul

Take `Y = X · W`, the operation that dominates everything.

| Symbol | Words |
|---|---|
| `X` | the input activations, shape `[T, k]` — `T` tokens, each a vector of `k` numbers |
| `W` | a weight matrix, shape `[k, n]` — the learned parameters of this layer |
| `Y` | the output, shape `[T, n]` — `T` tokens, each now a vector of `n` numbers |
| `T` | number of tokens in this matmul (batch × sequence length, flattened) |
| `k`, `n` | input and output widths of the layer |

Each output element `Y[t, j]` is a dot product of a length-`k` row of `X` with a length-`k`
column of `W`: `k` multiplies and `k−1` adds, which everyone rounds to `2k` FLOPs. There
are `T × n` output elements, so:

```
FLOPs_forward = 2 · T · k · n
```

`W` has exactly `k · n` parameters. Divide:

```
FLOPs_forward per parameter per token = 2 · T · k · n / (T · k · n) = 2
```

**Two FLOPs per parameter per token, forward.** That is the whole first half. It is a
statement about matrix multiplication, not about transformers, and it holds for any
layer whose parameters are used exactly once per token.

### The backward pass costs exactly twice the forward pass

Backpropagation through the same matmul must produce two gradients:

| Quantity | Words | Shapes | FLOPs |
|---|---|---|---|
| `dL/dX = dL/dY · Wᵀ` | how the loss changes with this layer's *input* — needed to keep propagating backwards | `[T,n] × [n,k]` | `2 · T · n · k` |
| `dL/dW = Xᵀ · dL/dY` | how the loss changes with this layer's *weights* — this is what the optimizer consumes | `[k,T] × [T,n]` | `2 · k · T · n` |

Two matmuls, each the same size as the forward one. So:

```
FLOPs_backward = 4 · T · k · n = 2 × FLOPs_forward
FLOPs_total    = 6 · T · k · n
```

**Six FLOPs per parameter per token.** Sum over every parameter in the model and every
token you train on:

```
C ≈ 6 · N · D
```

| Symbol | Words |
|---|---|
| `C` | total floating-point operations for the whole training run |
| `6` | 2 forward + 4 backward, per parameter, per token — derived above, not fitted |
| `N` | number of parameters that participate in a matmul once per token |
| `D` | number of training tokens (total, counting repeats if you repeat data) |

Read the same claim in production code:
`training/olmo-core/src/olmo_core/nn/lm_head.py:422` is the whole rule in three lines,
comment included — `# 6 FLOPs per parameter (2 ops * 3 for forward+backward)`.

Two footnotes that matter later. First, the very first layer does not strictly need
`dL/dX`, and neither does anything before a `detach` — a negligible correction at depth.
Second, this counts *no* activation recomputation. Turn on full activation checkpointing
and you re-run the forward pass, so the hardware does 8 FLOPs per parameter per token
while the MFU numerator still says 6. That gap is the MFU/HFU spread, and 8/6 = 1.33 is
where the "4/3" you see quoted comes from `[C]` (arXiv 2205.05198).

### Which parameters count as `N`

Not all parameters are used in a matmul.

- **Token embedding lookup** is a *gather*, not a matmul. Zero FLOPs. Its parameters
  should not be in `N` on account of the lookup.
- **Learned position embeddings** (`wpe` in nanoGPT) are also a gather. Zero FLOPs.
- **The output head** (`lm_head`, shape `[d, V]`) is a full matmul against every token.
  Its `V · d` parameters count fully.
- With **weight tying**, the token-embedding matrix *is* the output head. So its
  parameters count — once, via the head.

nanoGPT gets this exactly right and documents it in a docstring:
`training/nanogpt/model.py:150`. `get_num_params(non_embedding=True)` subtracts `wpe` and
*not* `wte`, "because due to the parameter sharing these params are actually used as
weights in the final layer." That is a four-line function that encodes the distinction
which confused the Kaplan and Chinchilla fits `[C]` (arXiv 2406.12907).

**At our scale this is not a footnote.** In the worked config below, the tied
embedding/head is 34% of all parameters. Get the convention wrong and your FLOP budget is
off by a third.

### What 6ND leaves out: attention has no parameters

`QKᵀ` and `softmax(·)V` are matmuls between two *activations*. No weights are involved,
so they contribute zero to `6N` — but they are not free, and they grow with sequence
length.

Per layer, per token, forward:

| Term | Words | FLOPs |
|---|---|---|
| `QKᵀ` | this token's query vector dotted against `T` cached key vectors, for each of `H` heads of width `Q` | `2 · H · Q · T` |
| `softmax(·)V` | the resulting `T` attention weights used to average `T` value vectors, per head | `2 · H · Q · T` |

Sum, multiply by `L` layers and by 3 for forward+backward:

```
FLOPs_attention per token = 12 · L · H · Q · T
```

| Symbol | Words |
|---|---|
| `L` | number of transformer layers |
| `H` | number of *query* heads per layer |
| `Q` | dimension of one head (`head_dim`) |
| `T` | sequence length — the number of positions each token can attend to |
| `12` | 2 matmuls × 2 FLOPs per multiply-add × 3 for forward+backward |

Note `H · Q = d_model` for a standard transformer, so this is `12 · L · d · T`.

The full per-token estimate, which is the PaLM formula and is what everyone actually uses:

```
flops_per_token = 6·N + 12·L·H·Q·T
```

Read it in two places we hold locally: `training/nanogpt/model.py:296` (used to print MFU
during training) and `training/nanogpt/transformer_sizing.ipynb:283` (derived and
sanity-checked against a term-by-term count at
`training/nanogpt/transformer_sizing.ipynb:227` and
`training/nanogpt/transformer_sizing.ipynb:247`).

**A convention warning, stated because it will otherwise silently inflate your numbers.**
This formula counts `T` keys per query. Causal masking means the average token attends to
about `T/2` positions, and a Flash-style kernel skips the masked half. So the standard
MFU numerator **over-counts attention by roughly 2×**, and that over-count grows as a
share of the total with sequence length. It is a convention, and everyone uses it; but at
long context it makes MFU look better than the machine is doing. Neither nanoGPT's
`flops()` nor olmo-core's `num_flops_per_token` applies a causal discount.

### When does the attention term matter?

Set the two terms equal and solve. For a standard transformer with 4× MLP expansion, the
trunk has `12·d²` parameters per layer (`4d²` attention projections + `8d²` MLP), so
`N = 12·L·d² + V·d` with a tied head.

```
12·L·d·T   =   6·(12·L·d² + V·d)
       T   =   6·d + V/(2·L)
```

| Symbol | Words |
|---|---|
| `T*` | the sequence length at which attention arithmetic equals *all* weight arithmetic |
| `d` | model width (`d_model`) |
| `V` | vocabulary size |
| `L` | layer count |

**`T* = 6·d_model + V/(2L)`.** Below it, 6ND is a decent approximation. Above it, 6ND is
wrong by more than a factor of two and you must carry the attention term. This is a
derived, checkable result and it explains why 6ND was fine in 2020 (GPT-3: `d = 12288`,
`T = 2048`, nowhere near) and is not fine for long-context work.

### Chinchilla, stated as math

Chinchilla's Approach 3 fits a parametric surface to observed final losses:

```
L(N, D) = E + A / N^α + B / D^β
```

| Symbol | Words | Fitted value |
|---|---|---|
| `L` | final training loss, in nats per token | — |
| `E` | irreducible loss — the entropy of the data; no model beats this | 1.69 |
| `A / N^α` | the penalty for having a finite number of parameters | `A` = 406.4, `α` = 0.34 |
| `B / D^β` | the penalty for having seen a finite number of tokens | `B` = 410.7, `β` = 0.28 |

Those constants are in code we hold: `training/nanogpt/scaling_laws.ipynb:462` defines
`L(N, D)` with exactly those five values on the following lines.

Now minimise `L` subject to the budget `6ND = C`. Substitute `D = C/(6N)`:

```
L(N) = E + A·N^(−α) + B·(C/6)^(−β)·N^(β)
dL/dN = −α·A·N^(−α−1) + β·B·(C/6)^(−β)·N^(β−1) = 0
α·A·N^(−α) = β·B·(C/6)^(−β)·N^(β)
N^(α+β) = (α·A)/(β·B) · (C/6)^β
```

```
N* = [ (α·A)/(β·B) ]^(1/(α+β)) · (C/6)^( β/(α+β) )
D* = (C/6) / N*
```

The exponent on compute is `β/(α+β) = 0.28/0.62 = 0.452` for parameters and
`α/(α+β) = 0.548` for tokens. Chinchilla's own Approaches 1 and 2 report both exponents
at ≈ 0.50, which is where "scale N and D together, 20 tokens per parameter" comes from.
**Approach 3 does not agree with Approaches 1 and 2, and this is not a subtlety — it is
the finding of the replication attempt** `[C]` (arXiv 2404.10102, Apr 2024), and it is
independently visible in the local notebook at
`training/nanogpt/scaling_laws.ipynb:569`, where the author reports a direct disagreement
with the paper's own table.

Work the coefficient once, because you will want it:

```
(α·A)/(β·B) = (0.34 × 406.4) / (0.28 × 410.7) = 138.18 / 115.00 = 1.2016
1.2016^(1/0.62) = 1.2016^1.6129 = 1.3448
N* = 1.3448 · (C/6)^0.4516
```

Apply it at Chinchilla's own budget, `C = 6 × 70e9 × 1.4e12 = 5.88e23`:

```
C/6 = 9.80e22
log10(9.80e22) = 22.991 ;  × 0.4516 = 10.383 ;  10^10.383 = 2.415e10
N* = 1.3448 × 2.415e10 = 3.25e10  =  32.5B parameters
D* = 9.80e22 / 3.25e10 = 3.02e12  =  3.02T tokens
D*/N* = 93 tokens per parameter
```

The paper's headline says 20. Its own Approach-3 constants, minimised, say 93 at that
budget. Both numbers are in the same paper. **Do not carry "20 tokens per parameter"
around as a fact; carry it around as one estimator's answer at one scale.**

And note *why* the ratio moves: `D*/N* ∝ C^((α−β)/(α+β)) = C^0.097`. The optimal ratio
grows slowly with compute. Which means — usefully for us — it is *smaller* at small
scale. Compute it at our budget in the next section.

---

## Why it matters for Proteus

### Budgeting our own run, end to end, with the arithmetic shown

Take a concrete rung. This is a `proteus-dense` baseline sized to sit in the middle of
our declared 20M–300M ablation box.

| Field | Value |
|---|---|
| `n_layers` (`L`) | 16 |
| `d_model` (`d`) | 512 |
| `n_heads` (`H`) | 8 |
| `head_dim` (`Q`) | 64 |
| MLP expansion | 4× (hidden 2048) |
| `vocab_size` (`V`) | 50,257 (GPT-2 BPE) |
| `seq_len` (`T`) | 1024 |
| embeddings | tied |

**Step one — count parameters.**

```
per layer:  attention projections  4·d²  = 4 × 262,144 = 1,048,576
            MLP up + down          8·d²  = 8 × 262,144 = 2,097,152
            two RMSNorms           2·d   =                   1,024
            ------------------------------------------------------
                                                        3,146,752
× 16 layers                                            50,348,032
final norm                                                    512
            ------------------------------------------------------
trunk                                                  50,348,544
tied embedding / lm_head   V·d = 50,257 × 512 =        25,731,584
            ------------------------------------------------------
N                                                      76,080,128   ≈ 76.1M
```

The head is 33.8% of `N`. At 300M it would be 8%. At 7B, under 1%. **Small models are
embedding-heavy, and that changes which conventions matter.**

**Step two — FLOPs per token.**

```
6·N                    = 6 × 76,080,128                    = 456,480,768
12·L·H·Q·T             = 12 × 16 × 8 × 64 × 1024           = 100,663,296
                                                             -----------
flops_per_token                                            = 557,144,064   ≈ 5.571e8
```

Attention is 18.1% of the total at `T = 1024`. The crossover formula predicts
`T* = 6×512 + 50,257/32 = 3072 + 1,570 = 4,642` — so at a sequence length of about 4.6k
this model spends as much arithmetic on attention as on all of its weights combined.

**Step three — pick `D`.** Chinchilla-style, `D = 20·N = 1,521,602,560 ≈ 1.52e9` tokens.

Cross-check against the Approach-3 optimum at this budget. Here `C_6ND = 6ND = 6.946e17`:

```
C/6 = 1.158e17 ;  log10 = 17.064 ;  × 0.4516 = 7.706 ;  10^7.706 = 5.086e7
N* = 1.3448 × 5.086e7 = 6.84e7  = 68.4M  ;  D* = 1.158e17/6.84e7 = 1.69e9
D*/N* = 24.8 tokens per parameter
```

So the parametric fit prefers 68M parameters and 1.69B tokens where `D = 20N` gives
76M/1.52B — an 11% difference in `N`, far inside the uncertainty on the fit itself.
**`D = 20N` is a defensible default at our scale**, and the reason is that
`D*/N* ∝ C^0.097` puts the ratio near 25 down here even though the same fit says 93 at
frontier compute.

**Step four — total compute.**

```
C_full = 557,144,064 × 1,521,602,560 = 8.478e17 FLOPs
C_6ND  = 456,480,768 × 1,521,602,560 = 6.946e17 FLOPs
```

**6ND under-budgets this run by 22%.** At `T = 1024`. On a 76M model. If you plan with
6ND alone and the sequence length goes up, the error grows without warning.

**Step five — wall clock. This is where the honesty is required.**

`[M]` Our measured bf16 GEMM throughput is **20.9 TFLOP/s at 8192³**
(`scripts/benchmark_gemm.py`, 2026-07-26, torch `2.12.0a0+rocm7.13.0a20260313`, HIP 7.2.0,
hipBLASLt configured; `ASSUMPTIONS.md → hipblaslt-config`). That figure is itself 63% of
the ~33 TFLOP/s cited for this silicon and the shortfall is unexplained
(`ASSUMPTIONS.md → gemm-throughput-below-reference`).

`[A]` **Sustained end-to-end training throughput of 6 TFLOP/s — 28.7% MFU against that
measured GEMM ceiling. This is an assumption, not a measurement. Confidence:
low-to-medium.** It is inherited from `research/notes/pretraining-recipes.md` §5 so the
two documents stay consistent, and it is the single most consequential unverified number
in the lab's planning. It is optimistic for a specific, nameable reason: our largest
GEMM at `d = 512` is nothing like 8192³, and small-`k` GEMMs run far below peak.

```
tokens/s = 6e12 / 5.571e8 = 10,770 tokens per second
run time = 8.478e17 / 6e12 = 141,292 s = 39.2 hours = 1.64 days
```

At a 65,536-token global batch (64 sequences × 1024) that is 23,220 optimizer steps at
6.1 s each.

**Step six — the sensitivity table, which is the actual deliverable.** Because the
throughput is `[A]`, the honest output of this exercise is not a number but a range:

| `[A]` sustained TFLOP/s | implied MFU vs 20.9 | one run | 3 seeds × 2 arms |
|---|---|---|---|
| 2 | 9.6% | 4.90 days | **29.4 days** |
| 4 | 19.1% | 2.45 days | 14.7 days |
| **6 (current assumption)** | **28.7%** | **1.64 days** | **9.8 days** |
| 8 | 38.3% | 1.23 days | 7.4 days |
| 10 | 47.8% | 0.98 days | 5.9 days |
| 20.9 (peak — unreachable) | 100% | 0.47 days | 2.8 days |

**A single properly-seeded two-arm comparison at 76M parameters costs somewhere between
6 and 29 days, and we do not know which.** That is a 5× planning uncertainty on the
lab's core unit of work. Closing it costs one instrumented nanoGPT run.

For contrast, the rented arithmetic, because G3 requires it: `[C]` H100 SXM bf16 dense
peak ≈ 989 TFLOP/s (NVIDIA datasheet; the 1,979 headline is the 2:4-sparsity figure —
and note olmo-core encodes exactly this correction at
`training/olmo-core/src/olmo_core/train/callbacks/speed_monitor.py:99`). `[A]` at 35% MFU
→ 346 TFLOP/s → `8.478e17 / 3.46e14 = 2,450 s = 0.68 h` per run; six runs ≈ 4.1 GPU-hours;
`[C]` at July-2026 on-demand pricing of $2.19–$3.20/GPU-hour (`pretraining-recipes.md` §5,
re-verify before any spend) that is **roughly $9–$13 for the whole comparison**.
`ASSUMPTIONS.md → cloud-budget-zero` is user-set and spending needs approval; the point
of the arithmetic is that the *information* is cheap and the *wall-clock* is not.

### The config surface is the FLOPs surface

Every ablation axis in `configs/` changes `N`, or the attention term, or both — and they
do not move together. That means **"matched budget" is ambiguous and Themis has to pick
a definition and record it.**

**Sliding-window attention (`proteus-swa-4to1` vs `proteus-dense`).** A windowed layer
attends to `min(w, T)` positions, so its attention FLOPs are capped. Read the
implementation: `training/olmo-core/src/olmo_core/nn/attention/__init__.py:806` is
literally `effective_seq_len = min(self.window_size, seq_len) if self.window_size else
seq_len`, feeding the `12·H·Q·seq` term on the next line. Work it for our rung with a
4:1 pattern (4 full layers, 12 windowed at `w = 256`):

| | `T = 1024` | `T = 8192` |
|---|---|---|
| dense attention FLOPs/token | 1.007e8 | 8.053e8 |
| 4:1 SWA attention FLOPs/token | 4.404e7 | 2.202e8 |
| dense total | 5.571e8 | 1.262e9 |
| SWA total | 5.005e8 | 6.767e8 |
| SWA cost relative to dense | 89.8% | **53.6%** |

At 8k context the SWA arm is 46% cheaper per token. So at matched parameters and matched
tokens it finishes in half the wall-clock — which means at matched *wall-clock* it would
see roughly twice the tokens. Three different "fair" comparisons, three different
answers, and the literature's hybrid-ratio results do not always say which one they ran.
This is the direct FLOPs-side complement to the capacity-side argument in
`research/memory/hybrid-architectures.md`: the hybrid ratio buys memory *and* compute,
and an experiment that credits it for both without saying so is measuring two things at
once.

**MoE (`proteus-moe-sigmoid`).** FLOPs scale with *active* parameters, memory with
*total*. The rule is one line:
`training/olmo-core/src/olmo_core/nn/moe/parallel_mlp.py:347` —
`return 6 * int(expert_params * self.top_k / self.num_experts)`. `[M]` Laguna-S routes 10
of 256 experts, an active fraction of 3.9%. So a matched-total-parameter MoE arm costs a
small fraction of the dense arm's FLOPs, and a matched-active-parameter MoE arm costs the
same FLOPs but many times the memory. **The house rule "matched param counts and token
budgets" does not resolve this.** Make it a config field and state it in the hypothesis
card.

**GQA (`n_kv` sweeps).** Changing `num_key_value_heads` changes the width of `k_proj` and
`v_proj`, so it changes `N` — a GQA sweep is not parameter-matched by default either.
`research/memory/kv-cache-mechanics.md` derives the memory side of that knob
(`2·L·n_kv·d_h·b` bytes per token, and decode arithmetic intensity `= 2G/b`); this module
is the training-compute side of the same config field.

**The consequence for the rig.** The single computed quantity every Proteus config must
expose is `num_flops_per_token(seq_len)`, aggregated the way olmo-core does it — walk the
modules and sum, rather than applying one global formula:
`training/olmo-core/src/olmo_core/nn/transformer/model.py:982`. Themis budgets against
`C`, the JSONL telemetry schema in `pretraining-recipes.md` already carries `tflops_est`,
and the pre-registration COST field becomes arithmetic rather than a guess.

### MFU, read honestly

```
MFU = (flops_per_token × tokens_per_second) / device_peak_flops_per_second
```

That is exactly the code:
`training/olmo-core/src/olmo_core/train/callbacks/speed_monitor.py:211`.

Three things to hold onto.

1. **The denominator is a choice.** nanoGPT hardcodes `flops_promised = 312e12`
   (`training/nanogpt/model.py:301`) — A100 bf16 peak. On the 8060S that number is
   meaningless and the printed MFU will read absurdly low even when everything is
   configured correctly. `[M]` Our denominator is 20.9e12, and it is a *measured* GEMM
   rather than a datasheet figure, which is arguably a better denominator and definitely
   a different one. Say which you used, every time.
2. **The numerator is a model, not a measurement.** It over-counts causal attention by
   ~2× and ignores recomputation. Both conventions inflate MFU relative to what the
   silicon actually did.
3. **MFU is the wrong metric for half of what this lab studies.** Decode attention has an
   arithmetic intensity of `2G/b` — 6 to 9 FLOPs per byte on our reference model against
   a `[M]` ridge point near 105 (`research/memory/kv-cache-mechanics.md` §3). A decode
   workload *should* show near-zero MFU; that is correct behaviour, not a defect. The
   complementary metric is MBU (model bandwidth utilization: bytes that must move ÷ peak
   bandwidth), and for a fixed workload MFU and MBU are locked together by the arithmetic
   intensity, so they are not independent knobs. There is no single agreed number, and
   `[C]` a 2026 argument exists that the whole compute/bandwidth balance of decoding
   hardware is mis-specified (arXiv 2607.13068, Jul 2026).

One more local detail worth knowing because it will show up on your own dashboard:
olmo-core emits `throughput/chinchilla multiple` — literally
`global_train_tokens_seen / (20 * num_params)` at
`training/olmo-core/src/olmo_core/train/callbacks/speed_monitor.py:176`. Chinchilla is a
live telemetry metric in production training code, hardcoded 20 and all.

---

## Read the code

All paths relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first.

**The rule, in its clearest form.**
- `training/olmo-core/src/olmo_core/nn/lm_head.py:422` — three lines and a comment:
  `6 * sum(p.numel() for p in self.parameters())`. This is `6N` with nothing else in the
  way. Note it discards `seq_len` on the line above: a pure weight term has no sequence
  dependence.

**The rule, decomposed the way a real trainer does it.**
- `training/olmo-core/src/olmo_core/nn/transformer/model.py:982` — `num_flops_per_token`
  sums over blocks and adds the head. Look at what it does *not* do: there is no global
  formula, because a hybrid model's layers do not agree. This is the shape Proteus's
  config surface has to take.
- `training/olmo-core/src/olmo_core/nn/attention/__init__.py:791` — the attention block's
  version. Read all 19 lines. Line 799 is the `6 × params` term; line 806 is the sliding
  window (`min(window_size, seq_len)`); line 807 is `12 · n_heads · head_dim ·
  effective_seq_len`. The docstring at 802–805 tells you honestly that Flash attention
  actually burns ~14× because of recomputation and they count the idealised 12× anyway —
  that is the MFU/HFU gap being taken deliberately, in a comment.
- `training/olmo-core/src/olmo_core/nn/moe/parallel_mlp.py:347` — `6 * expert_params *
  top_k / num_experts`. Active-parameter accounting in one line. Read the docstring
  above it: it admits the estimate *under*-counts for the non-dropless MoE path, i.e. the
  metric is knowingly wrong in a known direction.
- `training/olmo-core/src/olmo_core/nn/moe/moe.py:327` — the MoE block's total: router +
  shared expert + routed experts. The router is `6 × params` like anything else; it is
  the experts that break the rule.

**Where the FLOP count meets the clock.**
- `training/olmo-core/src/olmo_core/train/callbacks/speed_monitor.py:211` — MFU, one
  division. Read the comment two lines up: MFU is computed from FLOPs/sec so it stays
  correct under variable sequence length, which a tokens/sec metric would not.
- `training/olmo-core/src/olmo_core/train/callbacks/speed_monitor.py:99` —
  `dense_correction = 0.5`, with the comment explaining that vendor specs are the
  sparsity numbers. This one line is the whole argument for distrusting any MFU you did
  not compute yourself.
- `training/olmo-core/src/olmo_core/train/callbacks/speed_monitor.py:176` — the Chinchilla
  multiple as telemetry.
- `training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:551` —
  where the train module hands `num_flops_per_token` to the monitor; the plumbing that
  makes the metric available at all.

**The pedagogical version, with the derivation attached.**
- `training/nanogpt/transformer_sizing.ipynb:227` and `:247` — a term-by-term forward
  FLOP count (`attention/scores`, `attention/reduce`, `mlp/ffw1`, …) and then
  `backward_total = 2 * forward_total`, "use common estimate of bwd = 2*fwd". You derived
  that factor of 2 above; here is someone asserting it, which is how you will usually
  meet it.
- `training/nanogpt/transformer_sizing.ipynb:283` — the PaLM formula, compared against the
  term-by-term count in the next cell. The two agree closely, which is the sanity check.
- `training/nanogpt/transformer_sizing.ipynb:353` and `:363` — `6ND` used to estimate a
  real training run's duration, with a pointer to where the rule comes from.
- `training/nanogpt/model.py:150` — `get_num_params`, and the docstring explaining why
  `wpe` is subtracted and `wte` is not. Four lines that encode the Kaplan/Chinchilla
  bookkeeping distinction.
- `training/nanogpt/model.py:296` — the same PaLM formula inside `estimate_mfu`.
- `training/nanogpt/model.py:301` — `flops_promised = 312e12`. The hardcoded A100
  denominator. Anything printed as "MFU" by this file on our machine is wrong by
  construction; the exercises fix it.

**Chinchilla in executable form.**
- `training/nanogpt/scaling_laws.ipynb:462` — `def L(N, D)` with `E = 1.69`, `A = 406.4`,
  `B = 410.7`, `α = 0.34`, `β = 0.28` on the five lines below it. These are the constants
  you minimised by hand above.
- `training/nanogpt/scaling_laws.ipynb:569` — an independent reader reporting that the
  fit disagrees with the paper's own tables. Read this before you trust any single
  scaling-law number, including the ones in this module.

**The gate recipe you will time in the exercises.**
- `training/nanogpt/README.md:51` — the published target, 1.4697 validation loss on
  shakespeare_char.
- `training/nanogpt/README.md:85` — the CPU-fallback invocation and its own published
  target of 1.88.

---

## Exercises

All three run on the Z13. Activate with `. .\scripts\activate-lab.ps1` (dot-source it).
GPU is native Windows / gfx1151 / single device — no collectives, and keep any single
tensor under 32 GiB (`ASSUMPTIONS.md → large-tensor-fault-32gib`; the failure is a silent
hang at 0% CPU, not an exception). Nothing here comes close to that limit. Write scratch
scripts under `notebook/`; they are TDD-exempt only while they remain reproducible from
committed config.

### Exercise: reconcile three FLOP counts

**Difficulty: low-medium. Time: 60–90 minutes.** Produces three numbers and two ratios.

Instantiate nanoGPT's `GPT` class at the shakespeare_char config (6 layers, 6 heads, 384
channels, `block_size` 256) and compute the forward+backward FLOPs per token three ways:

1. **Analytically**, `6·N + 12·L·H·Q·T`, taking `N` from the model's own
   `get_num_params()` — note it prints its parameter count at construction
   (`training/nanogpt/model.py:148`), and note that the default `bias=True` in
   `GPTConfig` adds parameters you might not have predicted.
2. **Term by term**, porting the `flops()` function from
   `training/nanogpt/transformer_sizing.ipynb:215` to this config.
3. **Empirically**, with `torch.utils.flop_counter.FlopCounterMode` wrapped around one
   forward and one `loss.backward()` on a batch of random token ids. This dispatches at
   the ATen level and is device-agnostic, so it works identically on CPU and on gfx1151.

Report the two ratios (analytic ÷ counted, term-by-term ÷ counted). **Expect agreement
within a few percent.** Then answer two questions with the counter rather than from
memory:

- Does `FlopCounterMode` discount for causal masking? Run the model's attention with
  `is_causal=True` and `is_causal=False` and compare. Whatever you find is the
  convention you must state when you report MFU.
- What does the backward pass actually cost as a multiple of the forward? Count them
  separately. You derived 2×; confirm it, and explain any excess.

**CPU fallback:** this exercise is shape arithmetic. Run it entirely on
`device='cpu'` — it will take seconds, and the counted FLOPs are identical.

**Why it matters:** every wall-clock estimate in this lab is downstream of this number.
If your analytic count is 20% off, so is every schedule.

### Exercise: measure sustained throughput and replace the `[A]`

**Difficulty: medium. Time: 90 minutes including runs.** Produces the single number this
module says is missing, plus a plot.

Write a timing harness under `notebook/` that, for a given `(L, d, H, T, V)`, builds the
model, feeds it **random token ids** (no dataset, no tokenizer, no I/O — you are timing
compute, not the data path), runs 20 warmup steps and 100 timed forward+backward+step
iterations with bf16 autocast, and reports:

- tokens per second
- achieved TFLOP/s = `flops_per_token × tokens_per_second`
- MFU against the `[M]` 20.9e12 denominator — **and print the denominator you used, in
  the output line**

Sweep two axes and plot achieved TFLOP/s:

- model width `d ∈ {256, 384, 512, 768}` at fixed `T = 1024`
- sequence length `T ∈ {256, 512, 1024, 2048}` at fixed `d = 512`

**What to look for.** Achieved TFLOP/s should rise with `d` (bigger GEMMs, closer to the
8192³ regime the 20.9 was measured in) and rise with `T` (the attention term is
counted in your numerator, and larger `T` also means fewer, larger kernel launches). If
achieved throughput is flat across `d`, you are launch-bound or dataloader-bound, not
compute-bound — check by timing with the optimizer step removed.

**Expected result, stated as a prediction so it can be wrong:** `[A]` somewhere between
2 and 8 TFLOP/s sustained at these shapes, i.e. 10–40% MFU. If you measure above 10, be
suspicious of the harness before you are pleased — recheck that `torch.cuda.synchronize()`
brackets the timed region and that the loss is actually backpropagated.

**CPU fallback:** identical script with `device='cpu'`, `dtype=float32`, `L=4`, `d=128`,
`T=64` (nanoGPT's published CPU shape, `training/nanogpt/README.md:85`), 20 timed steps.
For the MFU denominator, first measure CPU peak by running `scripts/benchmark_gemm.py`
logic with `device='cpu'` at 2048³ — do not use a datasheet number. Expect a small
number of GFLOP/s and an MFU that is more meaningful than the GPU one, because the CPU
denominator you measured is honest.

**Fold the result back.** Whatever you measure goes into `ASSUMPTIONS.md` as a new `[M]`
row with the wheel version, and the `[A]` 6 TFLOP/s in this module and in
`pretraining-recipes.md` §5 gets superseded. That is the point of the exercise; the
curriculum is supposed to move.

### Exercise: reproduce the IsoFLOP fitting bias, then budget a real sweep

**Difficulty: medium. Time: 60 minutes for the core; +2–3 hours for the optional real
sweep.** Produces a plot showing a fitted optimum moving when nothing real has changed.

**Core (CPU, seconds to run).** Use the Chinchilla Approach-3 surface from
`training/nanogpt/scaling_laws.ipynb:462` as ground truth — you have a *noise-free*
loss function, which is the setup `[C]` arXiv 2603.22339 (Mar 2026) uses.

1. Fix a compute budget `C`. Generate an IsoFLOP slice: for a grid of `N` values, set
   `D = C/(6N)` and evaluate `L(N, D)`.
2. Compute the **true** optimum analytically with the closed form you derived above
   (`N* = 1.3448 · (C/6)^0.4516`).
3. Now do what the literature does: fit a **parabola in log N** to the grid points and
   take its vertex as the estimated optimum.
4. Plot estimated ÷ true optimum as you vary (a) the grid *width* — how many orders of
   magnitude of `N` you sample — and (b) the grid *centre* — offset the grid so the true
   optimum is off-centre.

**Expected result:** the parabola fit is biased even though the data has zero noise, and
the bias grows with grid width and with off-centre sampling. This is not a
signal-to-noise problem you can fix with more seeds. Then look at your plot and ask the
uncomfortable question: **a 20M–300M sweep on one GPU is a narrow, few-point,
almost-certainly-off-centre grid.** `[C]` 2603.22339 quantifies ~6.5% budget
misallocation on published Llama-3 IsoFLOP data; estimate the equivalent for your grid.

**Optional GPU extension (2–3 hours).** Run a real 5-point IsoFLOP slice at
`C = 1e16` — at `[A]` 6 TFLOP/s that is ~28 minutes per point, and at the measured
throughput from the previous exercise you will know the real cost before you start, which
is the entire skill this module teaches. Use shakespeare_char so the data path is a ~1MB
memmap and disappears into page cache. Fit the parabola to *your* five noisy points, and
compare the width of the resulting confidence interval to the systematic bias you
measured on noise-free data. Report which dominates.

**CPU fallback:** the core exercise is CPU-only by nature — it is curve fitting, and the
GPU contributes nothing. Say that out loud in your writeup; knowing which parts of an
experiment do not need the accelerator is half of a cost model.

---

## Self-check

1. You are told a model has 7B parameters and was trained on 2T tokens. Give the training
   compute in FLOPs, then state two specific reasons your answer could be off by more
   than 20% in either direction.

2. Your colleague reports 55% MFU on a long-context run and 12% MFU on a decode
   benchmark, and concludes the decode kernel is broken. What is wrong with the
   conclusion, and what would you measure instead?

3. For a model with `d_model = 1024`, `L = 24`, `V = 100,000`, at what sequence length do
   attention FLOPs equal all weight FLOPs? Show the substitution.

4. `proteus-swa-4to1` and `proteus-dense` have identical parameter counts and are trained
   on identical token counts at `T = 8192`. Are they a matched-budget comparison? Give
   three defensible definitions of "matched" and say which one you would pre-register.

5. Chinchilla's headline is 20 tokens per parameter. Its Approach-3 constants, minimised
   under `6ND = C`, give 93 at its own compute budget and about 25 at ours. Is one of
   these wrong? What does the discrepancy tell you about how to use the rule?

6. Someone proposes running the IsoFLOP sweep with three points instead of five to save
   two days. Beyond the obvious loss of statistical power, name the specific systematic
   problem this makes worse, and cite the source.

---

## What is still unsolved here

**Our own sustained throughput is unmeasured.** `[A]` 6 TFLOP/s, low-to-medium
confidence, is the load-bearing number in every schedule in this lab, and the sensitivity
table above shows it spans a 5× range of plausible answers. `research/notes/pretraining-recipes.md`
names the same figure as its riskiest assumption. One instrumented nanoGPT run closes it.
Until then, treat every wall-clock estimate in the curriculum as an order-of-magnitude
statement.

**The measured GEMM ceiling itself is unexplained.** `[M]` 20.9 TFLOP/s is 63% of the
~33 TFLOP/s `[C]` reported for this silicon (`ASSUMPTIONS.md → gemm-throughput-below-reference`).
Until that gap is understood, the MFU denominator on this machine is uncertain, which
means our MFU numbers are not comparable with anyone else's — only with each other.

**There is no accepted successor to Chinchilla.** The 2026 position is fragmentation into
conditional laws — per data quality, per optimizer `[C]` (arXiv 2602.07712, Feb 2026, via
`pretraining-recipes.md` §1), per MoE sparsity `[C]` (arXiv 2402.07871, Feb 2024; arXiv
2501.12370, Jan 2025) — not convergence. Anyone announcing "the new scaling law" is
announcing a conditional one.

**The fitting method is biased and the correction is new.** `[C]` arXiv 2603.22339 (Mar
2026) shows Approach 2's parabola fit is biased on noise-free data, from narrow grids,
off-centre sampling, and loss-surface asymmetry — *exactly* the conditions of a
one-GPU sweep. It recommends Approach 3 with variable projection. That recommendation is
four months old at the time of writing and has not been independently replicated.

**Extrapolation from small scale is the lab's riskiest structural assumption.**
`ASSUMPTIONS.md → ablation-scale-sufficient` is `[A]`, untested. Scaling laws are the only
instrument we have for arguing that a 30M result says anything about 300M, and `[C]` arXiv
2605.08541 (May 2026) argues the tokens-per-parameter *coverage* of your grid — not the
number of points — determines whether a fit extrapolates at all. Our declared box spans
D/N from 1.7 to 250 at its corners, which is either good coverage or four incomparable
experiments depending on how it is sampled.

**FLOP-counting conventions are not standardised, and the differences are large.**
Causal halving (~2× on the attention term), activation recomputation (MFU vs HFU, ~1.33×),
MoE capacity factor and dropped tokens, whether embeddings count in `N`, and whether the
denominator is a datasheet or a measurement. Every one of those is a factor between 1.3
and 2. There is no standard and no lint for it. Chiron's answer is to compute
`num_flops_per_token` from the config, log the denominator alongside the metric, and
state the convention in every writeup — which is a discipline, not a solution.

**MFU is the wrong metric for the thing this lab is actually about.** Memory-systems
research is bandwidth- and capacity-bound. `research/memory/kv-cache-mechanics.md` derives
decode arithmetic intensity as `2G/b` against a `[M]` ridge near 105 FLOP/byte — decode
*should* show single-digit MFU. MBU is the complement, but MFU and MBU are locked together
by arithmetic intensity for a fixed workload, and neither captures under-utilization caused
by memory *capacity* limits, which is precisely the regime the Z13 was chosen to study.
`[C]` arXiv 2607.13068 (Jul 2026) argues the compute/bandwidth balance of decode hardware
is itself mis-specified. There is no single number here yet, and inventing one would be a
contribution.

**Compute-optimal may be the wrong objective even for us.** `[C]` The inference-aware line
(arXiv 2401.00448, Dec 2023; arXiv 2501.18107, Jan 2025) and the test-time-scaling line
(arXiv 2604.01411, Apr 2026) both push toward heavy overtraining, and frontier practice
has moved far past 20 tokens per parameter. But our artifacts are ablation arms that get
measured once and deleted — we do not serve them. Whether a research lab whose output is a
*ranking* rather than a *model* should train compute-optimally, overtrain for a cleaner
signal, or undertrain for more arms per week, is genuinely open and nobody has written
it down. It is also directly testable here: run the same two-arm comparison at D/N of 5,
20, and 80, and check whether the ranking is stable. If it is not, half the published
small-scale ablation literature has a problem.

---

## Answers to the self-check

**1.** `C ≈ 6 × 7e9 × 2e12 = 8.4e22` FLOPs. Two reasons it could be off by >20%:
*(a)* the attention term is excluded. If that model trained at long sequence length, the
`12·L·H·Q·T` term is material — for `d = 4096`, `T* = 6×4096 + V/(2L)` is roughly 25k, so
at `T = 8192` attention adds about a third of the trunk cost. *(b)* `N` is ambiguous:
does 7B include the embedding and output head, and were they tied? At 7B that is a few
percent, but if the "7B" is a *MoE* total parameter count, the actual FLOPs are set by
active parameters and the estimate could be off by an order of magnitude
(`parallel_mlp.py:347`). Also acceptable: activation recomputation makes hardware FLOPs
~1.33× the model FLOPs, so a wall-clock estimate built from 6ND under-predicts.

**2.** MFU near zero is the *correct* reading for decode, not a symptom. Decode attention
has arithmetic intensity `2G/b` — 6 to 9 FLOP/byte on our reference model — against a
machine ridge near 105 FLOP/byte, so the FLOP units are supposed to be idle; the workload
is memory-bandwidth-bound by construction and no kernel fix changes that
(`research/memory/kv-cache-mechanics.md` §3). Measure MBU instead: bytes that must move
per token (weights once, plus the whole KV cache for that sequence) divided by achieved
time, against `[M]` ~200 GB/s. Also check the two MFU numbers used the same denominator
and the same causal convention before comparing them at all.

**3.** `T* = 6·d + V/(2L) = 6×1024 + 100,000/(2×24) = 6144 + 2083 = 8227` tokens. Below
about 8k, weights dominate; above it, attention does — and note the vocabulary term
matters here (25% of the answer) precisely because `L` is small.

**4.** No, not automatically. Three defensible definitions: *(i)* **matched parameters
and matched tokens** — what is described; SWA then costs 53.6% of dense's FLOPs at
`T = 8192`, so it also finishes in about half the wall-clock, and the comparison silently
credits SWA for being cheap. *(ii)* **matched compute `C`** — give the SWA arm ~1.87×
the tokens so both spend the same FLOPs; this answers "which architecture is better per
FLOP," which is the question a compute-constrained lab usually means. *(iii)* **matched
wall-clock** — closest to "what can I actually build this week," but it entangles the
result with hipBLASLt, kernel maturity, and this specific GPU, so it does not generalise.
Pre-register *(ii)* for architecture claims and *(i)* only when the claim is explicitly
about parameter efficiency — and say which in the hypothesis card, because the three give
different rankings.

**5.** Neither is wrong; they are answers to the same question from different estimators,
and the disagreement is a documented property of the paper (`[C]` arXiv 2404.10102, and
visible locally at `training/nanogpt/scaling_laws.ipynb:569`). The mechanism is that
Approaches 1 and 2 give `α ≈ β`, making `D/N` constant, while Approach 3's fitted
`α = 0.34 ≠ β = 0.28` makes `D*/N* ∝ C^0.097` — a ratio that *grows with compute*. Use
it as: 20 is a fine default at small scale (the fit says ~25 at our budget), it is
increasingly wrong as budgets grow, and any specific number should be recomputed from the
constants for *your* `C` rather than quoted.

**6.** It makes the **IsoFLOP parabola fitting bias** worse, and worse in a way more
seeds cannot repair, because the bias is systematic and present even on noise-free data
`[C]` (arXiv 2603.22339, Mar 2026). Three points force a parabola through three points
exactly — zero residual, no diagnostic, and the vertex is fully determined by where you
happened to place the grid, which is precisely the off-centre-sampling failure the paper
names. Related: `[C]` arXiv 2605.08541 (May 2026) argues extrapolation quality is set by
tokens-per-parameter coverage, and dropping points shrinks coverage.

---

## Sources

**arXiv, verified against the live API on 2026-07-26** (resolution proves the paper
exists and has the title shown; it does not prove it supports the claim beside it):

- `2001.08361` — *Scaling Laws for Neural Language Models* (2020-01-23). Kaplan et al.
- `2203.15556` — *Training Compute-Optimal Large Language Models* (2022-03-29). Chinchilla.
- `2204.02311` — *PaLM: Scaling Language Modeling with Pathways* (2022-04-05). Where MFU is defined; cited by olmo-core's implementation.
- `2205.05198` — *Reducing Activation Recomputation in Large Transformer Models* (2022-05-10). The MFU/HFU accounting.
- `2203.03466` — *Tensor Programs V* (2022-03-07). muP; why small-scale results transfer at all.
- `2305.13245` — *GQA* (2023-05-22). The `n_kv` knob that changes both `N` and KV bytes.
- `1911.02150` — *Fast Transformer Decoding: One Write-Head is All You Need* (2019-11-06). Why decode is bandwidth-bound.
- `2401.00448` — *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws* (2023-12-31).
- `2404.10102` — *Chinchilla Scaling: A replication attempt* (2024-04-15). Approach 3 inconsistent with Approaches 1 and 2.
- `2406.12907` — *Reconciling Kaplan and Chinchilla Scaling Laws* (2024-06-12). The discrepancy is largely non-embedding vs total parameter counting.
- `2402.07871` — *Scaling Laws for Fine-Grained Mixture of Experts* (2024-02-12).
- `2501.12370` — *Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models* (2025-01-21).
- `2501.18107` — *Scaling Inference-Efficient Language Models* (2025-01-30).
- `2410.05192` — *Understanding Warmup-Stable-Decay Learning Rates* (2024-10-07). Why a trunk-and-branch schedule changes the cost of a sweep.
- `2603.22339` — *Problems with Chinchilla Approach 2: Systematic Biases in IsoFLOP Parabola Fits* (2026-03-21).
- `2605.08541` — *Tokens-per-Parameter Coverage Is Critical for Robust LLM Scaling Law Extrapolation* (2026-05-08).
- `2607.13068` — *The Economics of AI Decoding Chips* (2026-07-10). The compute/capacity/bandwidth mismatch for decode.

**Verified by fetching the arXiv abstract page on 2026-07-26:**

- `2604.01411` — *Test-Time Scaling Makes Overtraining Compute-Optimal* (2026-04-01). Train-to-Test scaling laws; overtraining becomes optimal once inference sampling is budgeted.

**Cited via `research/notes/pretraining-recipes.md`, which verified them in its own pass:**

- `2602.07712` — *Towards Robust Scaling Laws for Optimizers* (Feb 2026). Per-optimizer laws are ill-conditioned.

**Local code, all pointers opened and confirmed 2026-07-26** (paths relative to
`research/reference/`):

- `training/nanogpt/model.py:148`, `:150`, `:296`, `:301`
- `training/nanogpt/transformer_sizing.ipynb:215`, `:227`, `:247`, `:283`, `:353`, `:363`
- `training/nanogpt/scaling_laws.ipynb:462`, `:569`
- `training/nanogpt/README.md:51`, `:85`
- `training/olmo-core/src/olmo_core/nn/lm_head.py:422`
- `training/olmo-core/src/olmo_core/nn/attention/__init__.py:791`, `:799`, `:806`, `:807`
- `training/olmo-core/src/olmo_core/nn/moe/moe.py:327`
- `training/olmo-core/src/olmo_core/nn/moe/parallel_mlp.py:347`
- `training/olmo-core/src/olmo_core/nn/transformer/model.py:979`, `:982`
- `training/olmo-core/src/olmo_core/train/callbacks/speed_monitor.py:99`, `:176`, `:211`
- `training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:551`

**Local measurements and registers:**

- `ASSUMPTIONS.md` — `hipblaslt-config` (`[M]` 20.9 TFLOP/s bf16 at 8192³),
  `gemm-throughput-below-reference` (63% of the cited figure, unexplained),
  `gpu-fast-tier-size` (`[M]` ≥62 GiB at ~200 GB/s), `large-tensor-fault-32gib`,
  `single-device-only`, `bf16-numerics-unproven`, `ablation-scale-sufficient`,
  `cloud-budget-zero`, `reference-model`.
- `ENVIRONMENT.md` — torch `2.12.0a0+rocm7.13.0a20260313`, HIP 7.2.0, driver
  `32.0.23033.5002`, gfx1151, native Windows.
- `scripts/benchmark_gemm.py`, `scripts/activate-lab.ps1`.
- `research/notes/pretraining-recipes.md` — §4 (Chinchilla and its critiques), §5 (the
  wall-clock and dollar table, and the `[A]` 6 TFLOP/s figure this module inherits and
  re-flags).
- `research/memory/kv-cache-mechanics.md` — §3 (decode arithmetic intensity `2G/b`, the
  `[M]` ~105 FLOP/byte ridge point).
- `research/memory/hybrid-architectures.md`, `research/memory/open-problems-ranked.md`.

**Non-arXiv:** NVIDIA H100 datasheet (bf16 dense peak ≈ 989 TFLOP/s; the 1,979 headline is
the 2:4-sparsity figure — corroborated in code by
`speed_monitor.py:99`'s `dense_correction = 0.5`). H100 on-demand pricing $2.19–$3.20/hr
surveyed 2026-07-26 in `pretraining-recipes.md`; prices move, re-verify before any spend.

---

## Decision / Riskiest assumption / Next test

**Decision.** Budget every Proteus run with `flops_per_token = 6N + 12·L·H·Q·T` computed
from the config (never bare `6ND`), make `num_flops_per_token(seq_len)` a required method
on every Proteus model, log the MFU denominator alongside every MFU value, and require
each hypothesis card to name which budget is matched across arms — parameters, tokens,
FLOPs, or wall-clock. Default `D = 20N` at our scale; it is within 11% of the Approach-3
optimum at our compute, which is well inside the fit's own uncertainty.

**Riskiest assumption.** The `[A]` 6 TFLOP/s sustained-throughput figure. It is not
measured, it is inherited, and it spans a 5× range of plausible values that turns a
two-arm three-seed comparison at 76M parameters into anything between 6 and 29 days.

**Next test.** The second exercise: a synthetic-batch timing harness across four widths
and four sequence lengths, reporting tokens/s and achieved TFLOP/s against the `[M]`
20.9e12 denominator. Half a day of work, produces an `[M]` row, and supersedes the `[A]`
in this module and in `pretraining-recipes.md` §5 simultaneously.
