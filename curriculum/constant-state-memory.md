---
title: Constant-state memory — SSMs, linear attention, and the fixed-size rolling aggregate
version: 1.0.0
date: 2026-07-26
track: C — Memory (the deep track)
prereqs: attention-variants-and-kv-cost, tensors-and-autograd
difficulty: 4/5 — the recurrence is four lines of code; what is hard is that its failure mode is invisible from the outside, and building the probe that makes it visible is the actual work
time: 3–4 h reading and working the algebra; 2–3 h for the exercises. Exercise A is 90 seconds of compute and an evening of staring at the output.
bridges_into: kv-compression-and-eviction, hybrid-architectures, agent-memory-systems
mirrors: research/memory/constant-state-memory.md
---

# Constant-state memory

**Difficulty and time, honestly.** Section 3 is the only part that will slow you down, and
only in one place: the step from "the read is a weighted blend" to "and therefore the
signal-to-interference ratio is √(d_k/n)". Everything after that is arithmetic. Budget
3–4 hours for sections 1–5, and protect the time for Exercise A — it produces the number
this module is built on, it runs on a laptop CPU in under two minutes, and it will
overturn at least one thing you believe after reading section 2.

---

## 1. What this module settles

**One:** every architecture in this family — Mamba-2/3, DeltaNet, Gated DeltaNet, RWKV-7,
mLSTM, Lightning Attention — is the *same* two-line recurrence over a fixed-size matrix,
differing only in what they are permitted to do to that matrix before writing, and once
you can write those two lines you can price and reason about all of them. **Two:** the
memory saving is real and enormous in bytes — `[M]` 24 MiB against 24 GiB at 128k context
on our reference model's shape, a factor of 1024 — but the *latency* saving is bounded by
Amdahl against the weight read, and the thing constant state actually buys on our machine
is the ability to reach the roofline ridge at long context, which a KV cache structurally
cannot. **Three:** the recall failure is not a cliff in the algebra — `[M]` it is a smooth
degradation whose closed form is `cos = 1/√(1 + (n−1)/d_k)`, matching measurement to within
0.01 across a 32× range of `n` — and the cliffs reported in the literature come from the
*decision rule* and from *training*, which is a different claim and needs to be argued
separately.

What it does **not** settle: whether constant state is the right target at all. Three
separate 2026 research lines abandon strict O(1) on purpose, and two well-resourced labs
published opposite production conclusions in the same year `[C]`. Section 8 leaves that
contested.

---

## 2. Theory in plain language

### 2.1 The bridge, stated properly

You have built the thing this module is about. Somewhere in your career you have replaced
an unbounded event log with a fixed-size rolling aggregate — a t-digest instead of every
latency sample, a HyperLogLog instead of a set of visitor IDs, an EWMA instead of the
window, a Bloom filter instead of the key set. The trade is always identical:

| Unbounded log | Fixed-size aggregate |
|---|---|
| Size grows with the number of events | Size fixed at construction |
| Every read scans (or indexes into) all of it | Every read touches the same bytes |
| Exact | Lossy |
| Replayable | Not replayable |
| Cost of a query grows with history | Cost of a query is constant |

A transformer's KV cache is the unbounded log. It keeps one exact `(key, value)` pair per
token forever, and decode re-scans all of it per generated token. The previous module
(`attention-variants-and-kv-cost.md`) priced that log exactly: `2 · L · n_kv · d_h · b`
bytes per token, and it is the *whole* cost story, because there is no hot subset and
every access is a full scan.

Constant-state models are the fixed-size aggregate. One matrix per layer, per sequence.
Its shape has no `T` in it. That is the entire idea and everything else is consequence.

**Where the analogy breaks, and this is the module.** Your aggregates have *bounded,
characterizable* error. A t-digest guarantees relative error at the tails. A HyperLogLog's
standard error is `1.04/√m`, a number you can put in a runbook. A Bloom filter's false
positive rate is a closed form in `(m, k, n)` and — crucially — it is **one-sided**: it
never lies about absence, only about presence, and you know which direction the lie goes.

A recurrent state has none of that. Its error is **data-dependent**: two sequences of the
same length, written with the same rule into the same state, can produce one perfect recall
and one confident fabrication, depending entirely on the geometric relationship between
keys that the model learned during pretraining and that you cannot inspect. There is no
`1.04/√m`. There is no one-sidedness. And there is no error term returned alongside the
answer — the read is a `d_v`-dimensional vector that looks exactly like a correct read,
whether or not it is one.

That is the sentence to carry through the rest of this module: **the aggregate is lossy in
a way that is data-dependent and invisible until a specific recall query fails.**

### 2.2 What replaced what

Softmax attention (`[C]` 1706.03762) computes, for query position `t`, a softmax over the
scores `q_t · k_j` for all `j ≤ t`, then returns the weighted sum of values. Two costs:
you keep all `t` pairs, and you read all `t` of them per step.

The escape was noticed early. If you replace `exp(q·k)` with a *kernel feature map*
`φ(q)·φ(k)`, the softmax's normalization becomes a sum you can also accumulate, and the
whole thing reassociates:

```
Σ_j φ(q_t)·φ(k_j) v_j   =   φ(q_t) · ( Σ_j φ(k_j) v_jᵀ )
```

The right-hand parenthesis does not depend on `t`. Accumulate it once, incrementally, and
you have removed the log. This is "linear attention," and the reason it did not immediately
win in 2020 is that the resulting model was bad at exactly the thing the cache was for.

Three things happened between 2023 and 2026 to make it competitive:

1. **Selectivity plus a hardware-aware scan** (Mamba, `[C]` 2312.00752, Dec 2023). Make the
   recurrence's parameters *input-dependent* so the model can decide what to keep, and write
   a kernel that keeps the state in SRAM. The systems half of that paper is as load-bearing
   as the math half; read it that way.
2. **State space duality** (Mamba-2 / SSD, `[C]` 2405.21060, May 2024). The recurrence and
   an attention-shaped matmul are two factorizations of the same object, so you can train it
   with the dense matmul and decode it with the recurrence. This is what made the family
   trainable at speed.
3. **The delta rule** (DeltaNet, `[C]` 2406.06484, Jun 2024; Gated DeltaNet, `[C]` 2412.06464,
   Dec 2024). Instead of *adding* new content, read back what the state currently returns for
   this key and write only the difference. This is the one genuinely new idea in the family
   since 2023, and section 3.6 is about why.

### 2.3 One recurrence, all the architectures

Write `t` for token index, `T` for sequence length. For one head:

- `k_t ∈ R^{d_k}` — the **key**: the address this token files its content under.
- `v_t ∈ R^{d_v}` — the **value**: the content being filed.
- `q_t ∈ R^{d_k}` — the **query**: the address being looked up this step.
- `S_t ∈ R^{d_v × d_k}` — the **state**. The entire memory of everything before token `t`.

```
S_t = M_t ⊙ S_{t-1}  +  β_t · v_t k_tᵀ        (write)
o_t = S_t q_t                                  (read)
```

`v_t k_tᵀ` is the **outer product**: the `d_v × d_k` matrix whose `(i,j)` entry is
`v_t[i] · k_t[j]`. It is rank 1 — the cheapest possible matrix meaning "content `v` at
address `k`". `β_t` is write strength. `M_t` is a transition operator applied to the
*existing* memory before the new write lands: the forget/edit step. (I write `⊙` loosely;
for the scalar and diagonal cases it is a multiply, for the Householder case it is a matrix
product on the key side. Section 3.6 makes it exact.)

What each architecture puts in `M_t`:

| Architecture | `M_t` | Reads as |
|---|---|---|
| Linear attention (kernel-feature style) | `I` | Accumulate forever, never forget |
| Lightning Attention `[C]` 2401.04658 | `I` or scalar decay | **Not a new recurrence** — an I/O-aware tiled *kernel* for the above |
| Mamba-2 / SSD `[C]` 2405.21060 | `a_t · I`, scalar `a_t = exp(−Δ_t·exp(A_log)) ∈ (0,1)` | One global TTL tick on the whole matrix |
| mLSTM (xLSTM) `[C]` 2405.04517 | `f_t · I`, exponential gating, plus a separate normalizer state | Same, but the read is normalized so accumulation cannot blow up |
| DeltaNet `[C]` 2406.06484 | `I − β_t k_t k_tᵀ` (Householder-type) | Read-before-write: erase this key's direction, then write |
| Gated DeltaNet `[C]` 2412.06464 | `α_t (I − β_t k_t k_tᵀ)` | Both: global decay *and* targeted overwrite |
| Gated DeltaNet-2 `[C]` 2605.22791 | channel-wise **erase** gate and channel-wise **write** gate, decoupled | `β` was doing two jobs; split them |
| RWKV-7 `[C]` 2503.14456 | diagonal-plus-rank-one, vector-valued decay | Per-channel TTL rather than one scalar |
| Mamba-3 `[C]` 2603.15569 | **complex-valued** scalar — decay *and rotation* | A state that can rotate can count; one that can only decay cannot |

The unified-notation comparison of the linear half (DeltaNet, Gated DeltaNet, KDA, Gated
DeltaNet-2) on expressivity, decay and throughput is `[C]` 2607.07953.

> **Systems bridge, and it is a good one for the *write* path.** The delta term is a
> compare-and-swap against whatever is currently resident: read the current value for this
> key, compute the difference, write the difference. `β = 1` is an unconditional overwrite,
> `β = 0` a no-op, in between a partial write. The decay `α_t` is a TTL tick applied to
> every entry at once.
>
> **Where it breaks on the *read* path — the whole module in one sentence.** Your CAS
> operates on an addressed slot and either succeeds or fails. This one operates on a
> *direction in a continuous vector space*, so writing key `k` partially clobbers every
> stored association whose key is not orthogonal to `k`, and the read never fails — it
> returns a blend and lets you decide what to believe.

### 2.4 The correction the code makes to the folk story

The mental model "gating = selective forgetting, delta = accumulation" is **backwards**,
and this is read out of the reference implementation rather than the paper.

Gated DeltaNet's decay is *one scalar per head applied to the entire `d_k × d_v` matrix*
(`architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:54`), so every
stored association is attenuated by exactly the same factor every step. There is no per-key
or per-channel TTL. This is not a kernel limitation — the production Triton kernel already
implements per-key and per-channel gates
(`architecture/flash-linear-attention/fla/ops/gated_delta_rule/fused_recurrent.py:138`
onward, `USE_GK` / `USE_GV`) and the layer simply does not pass them.

**So the gate is indiscriminate decay, and the delta term is the targeted erase.** Gated
DeltaNet-2 `[C]` 2605.22791 and RWKV-7 `[C]` 2503.14456 are both, in effect, fixes for
this. Exercise A measures the consequence directly, and finds that at a fixed decay the
gate makes an overwrite task *worse*, not better.

### 2.5 Where the aggregate analogy breaks — four places

**1. There is no miss signal.** A cache's single most useful property is that a miss is a
*typed event*: countable, alarmable, serviceable. A recurrent state returns a plausible
blend and cannot distinguish "I have this" from "I have three things that rhyme with this."
Hit rate is not merely unmeasured — it is undefined. Every observability instinct you own
assumes a distinguishable miss. Exercise A tries to synthesize one from the only quantity
available at decode time and `[M]` fails: AUROC 0.58–0.63 within a fixed context length,
and 0.46 — *worse than a coin* — when pooled across lengths.

**2. There is no backing store, therefore no promotion, no demotion, no fault path.** The
decay multiply *destroys* content rather than relocating it
(`architecture/mamba/mamba_ssm/ops/triton/ssd_state_passing.py:80` is the entire
inter-chunk recurrence: a decay and an add, in place). Token 5's contribution cannot be
recovered at token 5000, because it was not evicted to a slower tier, it was overwritten.
A KV cache is an append-only exact log you can re-scan; this is unreplayable.

**3. There are no addresses and no lines.** Keys are L2-normalized continuous vectors, so
every write smears across the whole matrix and a *similar* key partially clobbers a
neighbour's content. This is a fully-associative store with lossy superposition, not a
set-associative cache. There is no capacity miss, because there is no capacity check.

**4. "Constant memory" is a decode-time property only.** During training and prefill the
chunked scan materializes one state per chunk before the boundary pass runs, so activation
memory is `O(L / chunk_size)`, not `O(1)` — and that per-chunk state is computed in fp32
regardless of model dtype (`ssd_combined.py:375`, `states_in_fp32=True`). The
constant-size state is quietly the most numerically fragile part of the layer. Exercise C
prices this and finds `[M]` a 4 GiB prefill footprint against a 1 MiB decode state at
shipped defaults and 1M context — a factor of 4096.

A fifth, smaller one worth knowing so you do not mis-read a benchmark: **Lightning Attention
is not an architecture.** It is an I/O-aware tiled kernel — conventional attention for
intra-block terms, the linear-attention kernel trick for inter-block terms `[C]` 2401.04658
— standing in the same relation to linear attention that FlashAttention stands to
attention. Listing it beside Mamba-2 as a "variant" conflates a performance change with a
capability change. MiniMax-01 shipped it at 456B total / 45.9B active with 1M training
context `[C]` 2501.08313.

---

## 3. The math that actually matters

### 3.1 Symbols, every one translated

| Symbol | Reads as | Where it comes from |
|---|---|---|
| `T` | tokens in the sequence | runtime |
| `t` | token index, 1…`T` | runtime |
| `n` | number of distinct key–value pairs actually written | the data, not the config |
| `d_k` | width of a key/query vector, per head | config (`head_dim` / `d_state`) |
| `d_v` | width of a value vector, per head | config (`head_dim` / `headdim`) |
| `H` | heads per layer | config |
| `L` | layers | config |
| `B` | batch, i.e. concurrent sequences | runtime |
| `b` | **bytes per stored element** — 2 for bf16, 4 for fp32 | dtype |
| `S_t` | the state matrix, `d_v × d_k` | the thing under study |
| `M_t` | transition operator applied to `S_{t-1}` before the write | the architecture |
| `β_t` | write strength, scalar per head per token | learned (`b_proj`) |
| `α_t` | decay scalar per head per token, in (0,1] | learned (`a_proj` → `gate.py:46`) |
| `μ` | mutual coherence: `max_{i≠j} |k_i · k_j|` | the learned key geometry |
| `cl` | chunk length in the chunked scan | config (`chunk_size`) |
| `P`, `N` | Mamba-2's `headdim` and `d_state` — its `d_v` and `d_k` | config |

Note the collision I am keeping because both literatures use it: `L` is layers here and
also the name of the intra-chunk decay matrix in Mamba's own code. I will say which.

### 3.2 The write and the read, worked by hand

Take `d_k = d_v = 2`, pure accumulation (`M_t = I`, `β_t = 1`), two tokens, and keys that
happen to be orthogonal unit vectors.

```
k_1 = (1, 0)   v_1 = (3, 4)
k_2 = (0, 1)   v_2 = (5, 6)
```

The outer product `v_1 k_1ᵀ` is the 2×2 matrix whose `(i,j)` entry is `v_1[i]·k_1[j]`:

```
v_1 k_1ᵀ = | 3·1  3·0 |   =  | 3  0 |
           | 4·1  4·0 |      | 4  0 |

v_2 k_2ᵀ = | 5·0  5·1 |   =  | 0  5 |
           | 6·0  6·1 |      | 0  6 |

S_2 = sum      =  | 3  5 |
                  | 4  6 |
```

Read with `q = k_1 = (1,0)`: `S_2 q = (3·1 + 5·0, 4·1 + 6·0) = (3, 4) = v_1`. **Exact.**
Read with `q = k_2`: `(5, 6) = v_2`. Also exact. Two `(k,v)` pairs, four floats of storage,
perfect recall — and the KV cache would have needed eight floats for the same thing.

Now break the orthogonality. Let `k_2 = (0.6, 0.8)` (still unit norm, but `k_1 · k_2 = 0.6`):

```
S_2 = | 3  0 |  +  | 3.0  4.0 |  =  | 6.0  4.0 |
      | 4  0 |     | 3.6  4.8 |     | 7.6  4.8 |
```

Read with `q = k_1 = (1,0)`: `S_2 q = (6.0, 7.6)`. The true answer is `(3, 4)`. What you
got is `v_1 + 0.6·v_2 = (3,4) + (3, 3.6) = (6, 7.6)`. Exactly the prediction, and exactly
the failure mode: **you did not get a miss, you got `v_1` contaminated by 60% of `v_2`,
with no indication that anything happened.**

That is the whole thing. Everything below is bookkeeping on that one observation.

### 3.3 Why orthogonality is the pivot

With `M_t = I` and `β_t = 1`, `S_T = Σ_j v_j k_jᵀ`, so

```
o = S_T q = Σ_j v_j (k_jᵀ q)
```

— **every value ever written**, each scaled by the dot product between its key and your
query. If the keys were mutually orthogonal unit vectors and `q = k_5`, every term vanishes
except `v_5` and the read is exact. But you cannot pack more than `d_k` mutually orthogonal
directions into `R^{d_k}`. Past `n = d_k` pairs, non-orthogonality is *mandatory*, not a
modelling failure.

Softmax attention avoids this by not superposing at all: it keeps `t` separate `(k,v)`
pairs and applies a **sharpening nonlinearity** over the scores, which suppresses the
non-matching terms. Exact log plus sharp read, at `O(T)` bytes and `O(T)` reads per token.
The trade is stated exactly there, and it is the only honest way to state it.

### 3.4 The interference closed form — the one equation to memorize

Take `n` independent random unit keys in `R^{d_k}` and pure accumulation. Read with
`q = k_i`:

```
o = v_i · (k_i·k_i)  +  Σ_{j≠i} v_j (k_j·k_i)
  = v_i             +  interference
```

For independent random unit vectors in `R^{d_k}`, the cross terms `c_ij = k_j · k_i` have
mean 0 and variance `1/d_k`. The `v_j` are unit and near-orthogonal, so the interference
term's expected squared norm is the sum of the squared coefficients:

```
E ‖interference‖²  ≈  Σ_{j≠i} E[c_ij²]  =  (n − 1) / d_k
```

Signal power is 1. So:

```
signal-to-interference (power)      =  d_k / (n − 1)
signal-to-interference (amplitude)  =  √( d_k / (n − 1) )

cos(o, v_i)  =  1 / √( 1 + (n − 1)/d_k )
```

In words: **the fraction of the read that is the thing you asked for falls as the square
root of how full the state is.** At `n = d_k + 1` the interference is exactly as large as
the signal and `cos = 1/√2 = 0.707`. There is nothing special happening at that point — it
is a smooth curve — but that is the natural unit in which to denominate "full."

`[M]` **This is measured, and it holds.** `d_k = d_v = 64`, seeds {0,1,2}, fp64 state, CPU,
random unit keys and values (Exercise A):

| `n` | predicted `cos` | measured `cos` |
|---|---|---|
| 8 | 0.949 | 0.947 |
| 16 | 0.900 | 0.902 |
| 32 | 0.821 | 0.830 |
| 64 | 0.710 | 0.715 |
| 128 | 0.579 | 0.582 |
| 256 | 0.448 | 0.445 |

Agreement within 0.01 absolute across a 32× range of `n`. You now have a capacity-planning
formula for an associative memory, denominated in the same currency as everything else in
this curriculum.

### 3.5 The bound in three registers, and what each is worth

The closed form above is for *random* keys. A trained model does better, because it can
learn to spread keys out. How much better is bounded, and 2026 produced three statements of
the bound at different levels of rigour.

**Coding-theoretic — the useful one, and it is exact.** Let `G = K Kᵀ` be the Gram matrix of
the `n` unit keys: `G_ij = k_i·k_j`, diagonal all 1, and `rank(G) ≤ d_k` because the keys
live in `R^{d_k}`. Cauchy–Schwarz on the eigenvalues gives
`‖G‖_F² = Σλ_i² ≥ (Σλ_i)²/rank = n²/d_k`. The diagonal contributes exactly `n`, so the
off-diagonal mass — which *is* the interference power, summed over all `n` reads — obeys

```
Σ_{i≠j} c_ij²  ≥  n²/d_k − n            i.e., per read:   Σ_{j≠i} c_ij²  ≥  (n − d_k)/d_k
```

This is the first-order **Welch bound**, and equality holds exactly when the keys form a
tight frame. Substituting into §3.4 gives the hard ceiling on any key geometry whatsoever —
learned, designed, or lucky:

```
cos(o, v_i)  ≤  1 / √( 1 + (n − d_k)/d_k )  =  √( d_k / n )      for n > d_k
```

**Now compare with the random-key result.** Random gives `1/√(1 + (n−1)/d_k)`; the bound
gives `√(d_k/n)`. At `d_k = 64`: `n = 128` → 0.579 vs 0.707 (random is 82% of optimal);
`n = 256` → 0.448 vs 0.500 (90%). The gap *closes* as loading grows. Below capacity the
picture inverts completely — at `n ≤ d_k` an orthogonal key set has **zero** interference
while random keys already sit at 0.71–0.95.

The conclusion is sharp and it is the reason this module exists: **below `n = d_k`, key
geometry is everything and training can buy perfect recall; above `n = d_k`, the best
possible code beats random keys by ~10–20% and no amount of training removes the
interference floor.** Above capacity this stops being a machine-learning problem and becomes
an information-theory problem. KATA `[C]` 2607.17419 (Jul 2026) is the 2026 paper that takes
this seriously, recasting associative recall as spherical packing and characterizing capacity
by the Welch interference floor directly.

**Information-theoretic — treat with CAP-shaped care.** "The Impossibility Triangle of
Long-Context Modeling" `[C]` 2605.05066 (May 2026) proves via the Data Processing Inequality
and Fano's Inequality that no model achieves all three of: per-step compute independent of
sequence length, state size independent of sequence length, and recall of a number of facts
proportional to sequence length. *Caveat: single-author preprint, unreplicated.* The theorem
is only as useful as the fidelity of its model of a language model — which is exactly how
you should read CAP too.

**Empirical.** Zoology `[C]` 2312.04927 measured it in trained models: attention solves
multi-query associative recall at a constant model dimension of **64** across all tested
sequence lengths, while gated convolutions "do not achieve accuracy >0.9 unless `d ≥ N`" —
model dimension must scale with sequence length. That is the capacity wall in one inequality,
and it is the strongest transfer evidence in the field: gated-convolution architectures
underperform attention by **up to 2.1 perplexity points on the Pile**, and **82% of that gap**
localizes to tokens requiring associative recall — with a **70M Transformer beating a 1.4B
Hyena** (2.41 vs 3.43 ppl) on that slice.

### 3.6 The delta term, and what its eigenvalues buy

DeltaNet's `M_t = I − β_t k_t k_tᵀ` looks exotic; it is not. For a unit key `k`, `k kᵀ` is
the projection onto the direction `k`. So `I − β k kᵀ`:

- leaves every direction orthogonal to `k` **untouched** (eigenvalue 1, multiplicity `d_k − 1`),
- scales the `k` direction by `1 − β` (eigenvalue `1 − β`, multiplicity 1).

At `β = 1` it is an exact projection: it *deletes* the component of the state along `k`,
then the write puts `v_t kᵀ` back. That is your read-modify-write. At `β = 0` it is the
identity. At `β = 2` it is a Householder **reflection**, eigenvalue `−1`, and this matters:
negative eigenvalues are what unlock state tracking beyond the `TC⁰` limit `[C]` 2411.12537,
which is why
`architecture/flash-linear-attention/fla/ops/gated_delta_rule/fused_recurrent.py:127`
doubles `β` under an `ALLOW_NEG_EIGVAL` flag.

**Do not confuse state tracking with recall.** RWKV-7 sells recognizing all regular languages
`[C]` 2503.14456; Mamba-3 sells complex-valued state for state tracking `[C]` 2603.15569.
Neither is associative recall capacity, which is bounded by state *size* regardless of
eigenvalue range. The literature routinely presents both under "expressivity" and they are
different axes. Mamba-3 (SISO) scoring 12.2 on 4K multi-key retrieval while winning on
downstream average `[C]` 2603.15569 / 2605.22791 is the empirical form of that distinction.

### 3.7 Bytes — state versus cache, and the crossover nobody quotes

Per layer, per sequence, the recurrent state is:

```
state_bytes  =  H · d_k · d_v · b
```

Factor by factor: `H` heads each hold their own matrix; each matrix is `d_v × d_k`; `b`
bytes per element. **No `T`.** That absence is the entire product.

Gated DeltaNet-2 states its own numbers exactly: `H·d_k·d_v = 16·128·128 = 262,144` floats
per layer per batch element `[C]` 2605.22791. At bf16 that is **512 KiB per layer per
sequence** — at 1K context, at 1M context, identically.

Against the KV product from the previous module, `2 · n_kv · d_h · b` per token per layer.
For Laguna-S (`n_kv = 8`, `d_h = 128`, bf16) `[M]`: `2·8·128·2 = 4096 B = 4 KiB` per token
per layer.

**The crossover.** Set them equal:

```
T*  =  (H · d_k · d_v · b) / (2 · n_kv · d_h · b)  =  (H · d_k · d_v) / (2 · n_kv · d_h)
```

`b` cancels — the crossover is a pure shape ratio, independent of dtype. At the two shapes
above:

```
T*  =  (16 · 128 · 128) / (2 · 8 · 128)  =  262144 / 2048  =  128 tokens
```

`[M]` Verified by direct computation in Exercise B. **Below 128 tokens of context, the
"constant" state is *larger* than the KV cache it replaces.** Nobody puts this in an
abstract. It matters because it says constant state is a long-context technology and
nothing else — at chat-turn lengths it is a regression in bytes, and any benchmark run at
short context measures the wrong thing.

Whole-stack, at 48 layers:

| | Constant state | Laguna-S KV (all-global) | Ratio |
|---|---|---|---|
| any context | 24 MiB | — | — |
| 128 tokens | 24 MiB | 24 MiB | 1.0 |
| 8k | 24 MiB | 1.50 GiB | 64 |
| 128k | 24 MiB | **24.0 GiB** | **1024** |
| 1M | 24 MiB | 192 GiB | 8192 |

`[M]` The 24.0 GiB figure is `ASSUMPTIONS.md → kv-per-token-laguna`, computed from the
fetched config; the shipped 3:1 SWA/global interleave cuts real residency to 6.07 GiB at
128k, so 24 GiB is the all-global bound, not what you actually hold.

### 3.8 Bandwidth, Amdahl, and the argument that actually holds

Decode reads the entire cache once per generated token. Our measured fast tier runs at
`[M]` **199.9 GB/s out to ≥62 GiB** (`notebook/uma-carveout-controls-fast-tier.md`, single
run per arm). Arithmetic on measured inputs:

```
24 GiB  = 25.77 GB  ÷ 199.9 GB/s  =  128.9 ms   →   7.8 tok/s from cache traffic alone
24 MiB  = 0.0252 GB ÷ 199.9 GB/s  =    0.13 ms  →   7,900 tok/s
```

A 1024× ratio, and this is the number the constant-state literature is built on. **Now
apply Amdahl, because nobody in that literature does.** Laguna-S is 118B-A8.5B; the active
weights are 8.5B at bf16 = 17.0 GB, which must also stream once per token:

```
weights                  :  17.0 GB ÷ 199.9 GB/s  =   85.0 ms
+ all-global KV @128k    :                            128.9 ms   → 213.9 ms →  4.7 tok/s
+ shipped 3:1 KV @128k   :   6.07 GiB = 6.52 GB   =   32.6 ms    → 117.6 ms →  8.5 tok/s
+ 24 MiB constant state  :                              0.13 ms  →  85.1 ms → 11.8 tok/s
```

**Replacing the entire KV cache with a constant state buys 1.38× over the shipped hybrid at
128k, on our machine's bandwidth.** Not 1024×. If you had quoted the byte ratio as a speedup
you would be wrong by three orders of magnitude, and this is the single most common error in
reading these papers.

So where is the real win? **Capacity converts into batch, and batch converts into arithmetic
intensity.** Recall from the previous module that the weight-read half of decode has
arithmetic intensity exactly equal to batch size `B`, and our machine's roofline ridge is
`[M]` ≈105 FLOP/byte. Work it at a plausible Proteus arm (`[A]` medium confidence — 24
layers, `n_kv = 8`, `d_h = 64`, bf16, all-global, 300M params; the cheapest thing that would
move it is freezing an actual arm config):

- weights: 600 MB = 0.559 GiB, leaving 61.44 GiB of the `[M]` ≥62 GiB fast tier
- KV: `2·24·8·64·2 = 48 KiB/token`
- a matched constant-state arm at `H=8, d_k=d_v=64`: `8·64·64·2 = 64 KiB/layer` → **1.5 MiB
  per sequence**, and `T* = 32768/1024 = 32 tokens`

| Context | Max batch, KV | Max batch, constant state | Reaches the ridge (B ≥ 105)? |
|---|---|---|---|
| 4k | 327 | 41,943 | both |
| 12.8k | **105** | 41,943 | both, exactly at the edge |
| 32k | 40 | 41,943 | **KV: no.** state: yes |
| 128k | 10 | 41,943 | KV: no |
| 1M | 1 | 41,943 | KV: no |

The number to keep: **beyond ≈12,800 tokens of context, a 300M all-global model on this
machine cannot hold enough sequences to reach the roofline ridge, and is therefore
structurally stuck below peak no matter how well it is written.** Constant state removes
that ceiling entirely. That — not per-sequence latency — is the argument.

`[A]` High confidence in the algebra, zero measurements of end-to-end throughput on this
hardware. The cheapest test that would move it is a batched decode sweep at fixed context.

**And a Based-shaped caveat, because state size is a dial and not a property.** Based
`[C]` 2402.18668 shows you can traverse the recall-versus-state-size Pareto frontier by
varying the sliding-window size and the linear-attention feature dimension, recovering full
attention quality at one end and a small fixed state at the other, beating the strongest
sub-quadratic baselines on recall-intensive tasks by 6.22 accuracy points at 1.3B and
reaching 24× FlashAttention-2's generation throughput at 1024 generated tokens. Sparse Delta
Memory `[C]` 2607.07386 goes further and raises the state **~3 orders of magnitude at
identical FLOPs** — 8.0B state parameters against Gated DeltaNet's 2.2M at 8B scale — using
64 sparse reads and 64 sparse writes per step instead of a dense outer product. State size
and per-token compute are separable knobs. That is a memory hierarchy, spelled out.

### 3.9 "Constant" is decode-only: pricing the prefill

The chunked scan is the reason this family trains fast: split the sequence into fixed
`cl`-token chunks, compute each chunk's contribution independently with dense matmuls, then
run a short sequential recurrence over only the `L/cl` chunk *boundaries*. Structurally the
same shape as a segmented prefix-sum, or as replaying a checkpointed WAL where per-segment
deltas are computed in parallel and folded in strict order afterward.

Read the reference implementation (`ssd_minimal.py:34`) and count the two big allocations:

```
states     = [B, L/cl, H, P, N]     fp32  →  4·B·H·L·P·N / cl   bytes
L (segsum) = [B, H, L/cl, cl, cl]   fp32  →  4·B·H·L·cl         bytes
```

`P` is Mamba's `headdim` (its `d_v`), `N` its `d_state` (its `d_k`). These move in
**opposite** directions in `cl`. Total:

```
prefill_bytes(cl)  ≈  4 · B · H · L · ( P·N/cl  +  cl )
```

Differentiate and set to zero: the memory-optimal chunk length is

```
cl*  =  √(P · N)
```

At Mamba-2's shipped defaults (`headdim = 64` at `mamba2.py:45`, `d_state = 128` at
`mamba2.py:41`), `cl* = √8192 ≈ 90.5`. The shipped `chunk_size` is **256**
(`mamba2.py:59`) — 2.8× above the memory optimum, presumably bought for matmul efficiency.

`[M]` Measured on the Z13 (Exercise C, `B=1, H=32, L=8192, P=64, N=128`, fp32, gfx1151):
the U-curve reproduces with its flat minimum at `cl ∈ {64, 128}` — bracketing `cl* = 90.5` —
at 514 MiB peak, rising to 882 MiB at `cl=16` and 1,536 MiB at `cl=512`. Measured peak
exceeds the closed form by 322–354 MiB for `cl ≤ 128` (einsum intermediates) and by
1,008 MiB at `cl=512`; the closed form is a lower bound, and the *argmin* is what transfers.

**Important caveat, and it is the same lesson as `repeat_kv` in the previous module.** The
segsum term is a property of the *reference* implementation. The fused Triton path does not
materialize it — `_chunk_state_fwd` computes per-chunk states directly with
`states_in_fp32=True` (`ssd_combined.py:375`) and `_state_passing_fwd`
(`ssd_combined.py:379`) threads them. The `states` term is real in production; the segsum
term is not. So the closed form is the algorithm's memory, not the kernel's, and you must
say which one you are quoting.

The production term alone, at `B=1, H=32, P=64, N=128`:

| `L` | `cl=64` | `cl=256` | decode-time state |
|---|---|---|---|
| 32k | 0.50 GiB | 0.12 GiB | 1 MiB (fp32) |
| 128k | 2.00 GiB | 0.50 GiB | 1 MiB |
| 1M | 16.00 GiB | **4.00 GiB** | 1 MiB |

At shipped defaults and a 1M prefill you hold **4 GiB of fp32 chunk states against a 1 MiB
decode state** — a factor of 4096 — and it is a single tensor. Our `[M]`
`large-tensor-fault-32gib` says single tensors ≥32 GiB hang silently at 0 CPU. Solve for the
locus: `states = 31 GiB` when `L/cl = 31,744` chunks, i.e. `L ≈ 8.1M` at `cl=256`, `L ≈ 2.0M`
at `cl=64`, `L ≈ 1.0M` at `cl=32`. **A 1M-token prefill with `chunk_size=32` lands exactly on
a known silent-hang failure mode on this machine.** That is open question 8 of the survey
note, answered arithmetically; it now needs measuring rather than deriving.

---

## 4. Why this matters for Proteus and Mnemosyne

### 4.1 The config surface is the experimental surface

Every factor in §3.7 must be a first-class Proteus config field, because every one is an
ablation axis:

| Proteus config field | Factor | Ablation it enables |
|---|---|---|
| `state_heads`, `state_d_k`, `state_d_v` | `H · d_k · d_v` | state size at fixed parameter count — the matched-state-vs-matched-params dispute |
| `transition_kind` ∈ {`identity`, `scalar_decay`, `delta`, `gated_delta`} | `M_t` | attribution: which mechanism moves the cliff |
| `beta_source` ∈ {`fixed`, `learned`} | `β_t` | isolates the targeted-erase term |
| `alpha_source` ∈ {`one`, `fixed`, `learned`} | `α_t` | isolates the indiscriminate decay |
| `allow_negative_eigenvalues` | `β ∈ (0,2)` | state tracking, which is *not* recall |
| `state_dtype` | `b` | fp32 vs bf16 state, against `bf16-numerics-unproven` |
| `chunk_size` | `cl` | prefill memory, and the 32 GiB fault locus |
| `layer_types` (already exists) | which layers are recurrent vs attention | the hybrid ratio |

Note that `transition_kind`, `beta_source` and `alpha_source` are three fields where a
single "architecture name" would have been one. That is deliberate: **the survey's central
code-derived claim — that the gate is indiscriminate and the delta is the targeted erase —
is only testable if the two are separately switchable.** A config field per mechanism is the
difference between an ablation and a horse race.

### 4.2 The Mnemosyne-shaped question is the missing miss signal

`research/synthesis.md` concluded: *build the instrument, not another policy*. The reason
that lands hardest here is that a recurrent state is the **worst-instrumented memory system
in the curriculum**. A KV cache at least lets you ask "what would the answer have been if
entry `j` were present?" — that is the full-cache oracle the lab plans to build. A recurrent
state destroys the information, so there is no oracle-diff to compute and no entry to point
at.

So the question becomes: **can a usable confidence signal be synthesized from what remains?**
If yes, everything you know about fallback paths, hedged requests, degraded modes and
circuit breakers becomes applicable to a language model's memory, which it currently is not.
If no, that is worth knowing too, and it is a stronger negative result than the field has.

`[M]` Exercise A takes the first swing and misses. The only quantity available at decode
time without knowing the answer is the read norm `‖o‖`. Its AUROC as a predictor of "this
read is correct" is 0.576–0.631 within a fixed `n`, and **0.463 pooled across `n`** — worse
than a coin flip, because `‖o‖` rises with load while accuracy falls with load, so pooling
across the confounder inverts the signal. That is a Simpson's-paradox trap of exactly the
shape that ruins production dashboards, and it is the cheapest possible warning about
aggregating a memory-health metric across request sizes.

The negative result is narrow and should be read narrowly: it says *the unnormalized read
norm of an untrained delta-rule state is not a confidence signal*. It does not say a learned
probe on the full state cannot be one. That distinction is the experiment.

### 4.3 The paper closest in shape to a Mnemosyne contribution

HOLA `[C]` 2607.02303 (Jul 2026) pairs a delta-rule compressive state with a **bounded exact
KV cache**, admitting tokens by residual magnitude `β·‖e‖` — that is, admitting exactly the
tokens the state failed to absorb. At 340M / 15B SlimPajama tokens: Wikitext ppl 27.32 →
22.92 (−16.1%), below a full-attention Transformer++ at 26.88, with RULER needle recall
holding to 32k (16× training length). *Single author, unreplicated.*

Read the admission rule as a systems engineer: it is a **write-through cache with an
admission policy keyed on compression residual**. The signal that decides admission is the
same residual the delta rule already computes at `naive.py:56` — it is free. That is the
most Mnemosyne-shaped idea in the literature, and note what it concedes: a strictly constant
state is the wrong constraint. So do Log-Linear Attention `[C]` 2506.04761 (logarithmically
growing state), Sparse Delta Memory `[C]` 2607.07386 and Sparse State Expansion `[C]`
2507.16577 (huge state, sparsely addressed). **The research question has quietly moved from
"can we do without a KV cache" to "what is the right size and admission policy for the exact
tier."** That is a question Mnemosyne is built to answer and Proteus is built to host.

### 4.4 What not to build

The synthesis parked eviction-policy design: ~30 policies exist with no dominance result.
The same logic applies here with more force. Do not add an architecture to this table. The
useful contribution is the *instrument* that separates the mechanisms already in it — and
§4.1's three-field split is what makes that possible at all.

---

## 5. Read the code

All paths relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. Line numbers are pinned to the revisions in
`PROVENANCE.md`. Read in the order given.

### 5.1 The entire recurrence, four lines

| Where | What to look at, and why |
|---|---|
| `architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:54` | `h = h.clone() * g[:, :, i].exp()[..., None, None]` — the decay. **One scalar multiplies the entire `K×V` state.** Every stored association attenuated by the same factor. This is `M_t = α_t I`, and it is the whole "gated" half. |
| `architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:56` | `b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)` — the **read-before-write**. `h^T k` is what the state currently returns for this key; subtracting it means the write carries only the residual. Note the ordering: decay happened first, so the correction is measured against the *post-decay* state. |
| `architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:58` | `h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)` — the write. A rank-1 outer product. **The only place new information ever enters memory.** |
| `architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:59` | `o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)` — the read. One matrix–vector product against the whole state. There is no branch here, no lookup, no miss. |

Four lines. Print them and pin them above your desk; the rest of this family is variations.

### 5.2 Where the scalars come from, and the gates nobody passes

| Where | What to look at, and why |
|---|---|
| `architecture/flash-linear-attention/fla/layers/gated_deltanet.py:149` | `self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)` — `β`, the write strength, is **one scalar per head**, not per channel. Output width is `num_v_heads`, not `head_dim`. The sibling `a_proj` at `:148` produces the decay input the same way. |
| `architecture/flash-linear-attention/fla/ops/gated_delta_rule/gate.py:46` | `(-A_log.float().exp() * F.softplus(g)).to(output_dtype)` — always ≤ 0, so `exp(g) ∈ (0,1]`. A learned per-head, per-token TTL tick rather than a fixed half-life. |
| `architecture/flash-linear-attention/fla/ops/gated_delta_rule/fused_recurrent.py:136` | `b_h *= exp(b_g)` — the same global decay, in the production Triton decode kernel. The naive version is faithful, not a simplification. |
| `architecture/flash-linear-attention/fla/ops/gated_delta_rule/fused_recurrent.py:138` | `if USE_GK:` — **the kernel already implements per-key gates**, and `USE_GV` at `:145` per-channel value gates. Gated DeltaNet does not pass either. This is the code that makes §2.4's correction a fact rather than an opinion. |
| `architecture/flash-linear-attention/fla/ops/gated_delta_rule/fused_recurrent.py:153` | `b_v = b_beta * (b_v - tl.sum(b_h * b_k[None, :], 1))` — the delta step as the decode kernel runs it, state in registers. |
| `architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:152` | `S = S * decay[:, :, i, -1, None, None].exp() + ...` — the **chunked training form** of the identical recurrence: 64 sequential token writes collapse into one decayed state plus one batched matmul. Write-combining, with intra-chunk corrections precomputed. Compare against `:54–58` and satisfy yourself they are the same operator. |

### 5.3 Mamba-2: proof that the cache is constant, and where it stops being constant

| Where | What to look at, and why |
|---|---|
| `architecture/mamba/mamba_ssm/modules/ssd_minimal.py:34` | `ssd_minimal_discrete` — a ~40-line pure-PyTorch reimplementation of the chunked algorithm, numbered by step. **Read this before any Triton.** The two allocations Exercise C prices are on lines 54 and 60. |
| `architecture/mamba/mamba_ssm/modules/mamba2.py:352` | `ssm_state = torch.zeros(batch_size, self.nheads, self.headdim, self.d_state, ...)` — the inference cache allocation. **No `seqlen` dimension.** Contrast with a KV cache allocated per `max_seqlen`; this one line is the entire claim of the family. |
| `architecture/mamba/mamba_ssm/modules/mamba2.py:317` | `ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)` — one autoregressive step in plain PyTorch. Decay in place, accumulate in place. Note it is a `copy_`: the previous state is **gone**, not versioned, not journalled. |
| `architecture/mamba/mamba_ssm/ops/triton/ssd_state_passing.py:72` | `for c in range(nchunks):` — the "parallel scan" is a plain **serial** loop over chunk boundaries. All the time-axis parallelism is in the intra-chunk matmuls. |
| `architecture/mamba/mamba_ssm/ops/triton/ssd_state_passing.py:80` | `states = scale * states + new_states` — the entire inter-chunk recurrence. A decay and an add. **This is the line where the analogy to a storage tier dies:** the carry is destructively overwritten, so there is nothing to promote, demote, or fault in. |
| `architecture/mamba/mamba_ssm/ops/triton/ssd_combined.py:375` | `_chunk_state_fwd(..., states_in_fp32=True)` — the per-chunk states are fp32 **regardless of model dtype**. The constant-size state is the most numerically fragile part of the layer, which matters directly against `ASSUMPTIONS.md → bf16-numerics-unproven`. |
| `architecture/mamba/mamba_ssm/ops/triton/ssd_combined.py:379` | `_state_passing_fwd(...)` — the seam: per-chunk states computed in parallel on `:375` are handed here to be stitched sequentially. `initial_states` / `return_final_states` on the public entry point (`architecture/mamba/mamba_ssm/ops/triton/ssd_combined.py:628`) are the cross-request prefix-reuse hook — the SSM analogue of prefix KV reuse, except it is one small tensor instead of a block table. |

### 5.4 The hybrid escape hatch, in the smallest implementation

| Where | What to look at, and why |
|---|---|
| `architecture/samba/lit_gpt/model.py:323` | `self.use_mamba = layer_idx % config.mb_per_layer == 0 if config.mb_per_layer > 0 else False` — the entire interleaving pattern is one modulo, recomputed per block rather than stored, the way a striping function derives a device from a block number. Every hybrid-ratio question this lab asks is a question about this expression. |
| `architecture/samba/lit_gpt/config.py:33` | `full_per_layer: int = 1000000` — the default that makes **every** Samba attention layer windowed and none global. The papers' "attention for precise recall" story therefore covers only the last 2048 tokens; all longer-range carry is the recurrent state alone. Check a config before you believe a diagram. |

### 5.5 The contrast — what a memory with an allocator looks like

| Where | What to look at, and why |
|---|---|
| `memory/vllm/vllm/v1/core/block_pool.py:647` | `def get_new_blocks` — vLLM's KV allocator. Read it purely for what a recurrent state does **not** have: a free list, a refcount, a hash, a place where allocation can be refused. When this returns nothing the request is preempted; there is still no *fault*, but there is at least an event. A recurrent state has no equivalent of even this line. |

---

## 6. Exercises

All three run on the Z13. Activate first, in PowerShell, dot-sourced:

```powershell
. .\scripts\activate-lab.ps1
```

Standing hardware caveats from `ASSUMPTIONS.md`: single tensors **≥32 GiB hang** the GPU
silently at 0% CPU (`large-tensor-fault-32gib`); keep every buffer under 31 GiB. bf16
numerics on gfx1151 are **untested** (`bf16-numerics-unproven`). The Hardware Validation
Gate has not run, so nothing measured here is evidence by house standard — these are
instrument-shakedown runs and should be labelled as such in any notebook entry.

Write scratch scripts under `notebook/`. Exercise A should migrate into Mnemosyne with tests
when it is reused, because it is the seed of the capacity-probe instrument.

---

### Exercise A — the capacity of a state, with no training at all

**Goal:** separate the *algebra* of superposition from anything a trained model could learn
to do about it, and get a curve you can check against a closed form. This is the exercise
that produces the module's headline numbers, and it is the one that will change your mind.

**Hardware:** none — CPU, fp64, no torch GPU path. **Runtime:** ~90 s.
**Difficulty:** 2/5 to run, 4/5 to interpret.

```python
"""Capacity of a fixed-size recurrent state, measured WITHOUT training.

Write n (key, value) pairs into one d_k x d_v state matrix under three write
rules, then read every key back and ask whether the vector that comes out is
nearer to the right stored value than to any other. Nothing is learned.
"""
import torch

D_K = D_V = 64
SEEDS = [0, 1, 2]
N_GRID = [8, 16, 32, 48, 56, 64, 72, 96, 128, 192, 256]

def make_pairs(n, gen, orthogonal=False):
    v = torch.randn(n, D_V, generator=gen, dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True)
    if orthogonal:
        q, _ = torch.linalg.qr(torch.randn(D_K, D_K, generator=gen, dtype=torch.float64))
        return q[:n].contiguous(), v
    k = torch.randn(n, D_K, generator=gen, dtype=torch.float64)
    return k / k.norm(dim=-1, keepdim=True), v

def write_linear(k, v, dtype):                       # M_t = I
    h = torch.zeros(D_K, D_V, dtype=dtype)
    for i in range(k.shape[0]):
        h = h + torch.outer(k[i].to(dtype), v[i].to(dtype))
    return h

def write_decay(k, v, dtype, alpha=0.98):            # M_t = alpha * I
    h = torch.zeros(D_K, D_V, dtype=dtype)
    a = torch.tensor(alpha, dtype=dtype)
    for i in range(k.shape[0]):
        h = h * a + torch.outer(k[i].to(dtype), v[i].to(dtype))
    return h

def write_delta(k, v, dtype, beta=1.0):              # M_t = I - beta k k^T
    h = torch.zeros(D_K, D_V, dtype=dtype)
    b = torch.tensor(beta, dtype=dtype)
    for i in range(k.shape[0]):
        ki, vi = k[i].to(dtype), v[i].to(dtype)
        resid = vi - h.t() @ ki                      # naive.py:56
        h = h + torch.outer(ki, b * resid)           # naive.py:58
    return h

def read_and_score(h, k, v):
    o_raw = (h.t().to(torch.float64) @ k.t()).t()    # naive.py:59
    norms = o_raw.norm(dim=-1)
    o = o_raw / norms.clamp_min(1e-30)[:, None]
    sims = o @ v.t()
    correct = sims.argmax(dim=-1) == torch.arange(k.shape[0])
    return dict(acc=correct.double().mean().item(), cos=sims.diagonal().mean().item(),
                correct=correct, read_norm=norms)

def softmax_kv(k, v, scale):                          # the exact log, for control
    w = torch.softmax((k @ k.t()) * scale, dim=-1)
    o = w @ v
    o = o / o.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return ((o @ v.t()).argmax(dim=-1) == torch.arange(k.shape[0])).double().mean().item()
```

Then five reports. Each is a few lines of loop over `N_GRID` and `SEEDS`:

1. **The capacity curve.** Accuracy for `delta` / `decay0.98` / `linear`, plus the softmax
   control at scale `d_k**-0.5` *and* at a sharpened scale (`64.0`). Also print
   `delta_cos`.
2. **The same with exactly orthogonal keys** (`orthogonal=True`, `n ≤ d_k`).
3. **The overwrite test.** Build a stream of `2n` writes where each of `n` keys appears
   twice with different values, at *random* positions. Score whether the *later* value is
   the one that comes back, against a candidate set containing both.
4. **State dtype.** Same delta-rule curve with the state held in fp64 / fp32 / bf16.
5. **Recency and the miss signal.** At `n = 2·d_k`, accuracy bucketed by write order. Then
   AUROC of `‖o‖` (the raw read norm — the *only* quantity available at decode time that
   does not require knowing the answer) as a predictor of correctness, computed both pooled
   across `n` and within a single `n`.

**Deliverables — six numbers and one plot.** The plot is accuracy against `n/d_k` with one
line per write rule. The numbers, with what I measured `[M]` so you can check your harness
(`d_k = d_v = 64`, seeds {0,1,2}, fp64 state unless stated, CPU, torch
`2.12.0a0+rocm7.13.0a20260313`, fully deterministic — reruns are bit-identical):

| Report | What I got |
|---|---|
| delta accuracy at `n = d_k = 64` | **0.901**; at `n=32` 1.000, at `n=128` 0.526, at `n=256` 0.266 |
| pure accumulation at `n = 128` | **0.982** — *better than delta*, see below |
| orthogonal keys, both rules, all `n ≤ d_k` | **1.000** |
| interleaved overwrite at `n = 64` | delta **0.875**, decay 0.734, gated-delta 0.583, linear 0.469 |
| bf16 vs fp64 state | identical to 3 d.p. at every `n` except 0.901→0.906 (`n=64`) and 0.266→0.267 (`n=256`) |
| AUROC(`‖o‖` → correct) | **0.463 pooled**, 0.576–0.631 within a fixed `n` |

One extra check worth two lines of code: for pure accumulation, compare `delta_cos`'s
sibling — the measured `cos` for the `linear` rule — against **both** §3.4's random-key
prediction `1/√(1+(n−1)/d_k)` and §3.5's hard ceiling `√(d_k/n)`. `[M]` At `n = 128` I
measured 0.582 against a prediction of 0.579 and a ceiling of 0.707; at `n = 256`, 0.445
against 0.448 and 0.500. If your measured value ever *exceeds* `√(d_k/n)`, you have a bug —
that quantity is bounded by linear algebra, not by the model.

**Five things to argue with, because each one is a trap.**

1. **The softmax control at the standard `1/√d_k` scale scores 0.167 at `n=8` and 0.005 at
   `n=256` — near chance.** At a sharpened scale it scores **1.000 at every `n`**. You have
   just reproduced, from first principles, the exact artifact the survey note flags in
   Variational Linear Attention's Table 11 `[C]` 2605.11196, where a softmax control scores
   0.152 on a task Zoology shows attention solving essentially perfectly. The mechanism:
   with random unit keys, `k_i·k_i/√d_k = 1/8` and `k_i·k_j/√d_k ~ N(0, 1/(64·64))`, so the
   softmax is nearly uniform and the read is a mean of all values. Real attention has
   *learned* query and key projections that sharpen this. **A control that has not been
   trained is not a control.** Write that in the notebook entry.
2. **Pure accumulation beats the delta rule on write-once random keys.** This is not a bug.
   At `β = 1`, `I − k kᵀ` deletes the whole `k` direction, and every previously stored
   association whose key is not orthogonal to `k` loses content in the process. Plain
   accumulation never destroys anything; it only adds noise. So on a task where each key is
   written exactly once, the delta rule pays for a capability it is not using.
3. **Report 3 flips the ranking completely, and that is where the delta term earns its
   keep.** Rewrite each key once and pure accumulation collapses to 0.469 — a coin flip
   between the stale and the fresh value, which is exactly what "add both" predicts — while
   delta holds 0.875. The compare-and-swap story is confirmed by measurement.
4. **In report 3, adding the gate makes it worse** (0.583 vs 0.875 at `n=64`). At `α=0.98`
   over 128 steps, old content is attenuated 13×, and the gate cannot tell which content is
   stale. This is §2.4's code-derived claim as a number. **Caveat, and state it:** `α` here
   is hand-fixed, whereas a real Gated DeltaNet learns it per head per token and would
   presumably learn `α ≈ 1` on a task like this. What the measurement shows is the
   *mechanism*, not a prediction about a trained model.
5. **The curve has no cliff.** Delta accuracy goes 1.000 → 0.965 → 0.901 → 0.526 → 0.266
   across `n/d_k` = 0.5 → 0.75 → 1.0 → 2.0 → 4.0. Smooth. The cliffs in the literature
   (VLA's DeltaNet column drops 0.965 → 0.009 between `n=8` and `n=16` at `d_h=32`
   `[C]` 2605.11196) come from somewhere else — see the self-check.

**CPU/GPU note.** There is no GPU version and there should not be. `d_k = 64` and `n ≤ 256`
is a rounding error of compute; putting it on the GPU adds launch overhead and a numerics
question you do not want in a measurement whose whole point is exactness.

---

### Exercise B — the two crossovers, and why they are 8× apart

**Goal:** measure the constant-state advantage as a *decode step*, and find out how much of
the byte-ratio argument survives contact with a kernel.

**Hardware:** one gfx1151 GPU, native Windows; falls back to CPU/fp32 automatically.
**Runtime:** 2–3 min GPU, 5–10 min CPU. **Difficulty:** 2/5.

```python
"""Per-layer decode step: fixed-size recurrent state vs growing KV cache."""
import time, torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT  = torch.bfloat16 if DEV == "cuda" else torch.float32
BPE = torch.finfo(DT).bits // 8

H, DK, DV   = 16, 128, 128     # Gated DeltaNet-2 shaped state, one layer
N_KV, D_H, G = 8, 128, 6       # Laguna global layer: 8 KV heads, G = 6
BATCH, ITERS = 1, 30
T_GRID = [128, 512, 2048, 8192, 32768, 131072]

state_bytes = H * DK * DV * BPE
kv_per_token = 2 * N_KV * D_H * BPE
print("BYTES CROSSOVER T* =", state_bytes // kv_per_token, "tokens")

def recurrent_step(h, k, v, q, g, beta):
    h = h * g
    resid = v - torch.einsum("bhkv,bhk->bhv", h, k)
    h = h + torch.einsum("bhk,bhv->bhkv", k, beta * resid)
    return h, torch.einsum("bhkv,bhk->bhv", h, q)

def attention_step(K, V, q):
    s = torch.einsum("bhtd,bhgd->bhgt", K, q) * (D_H ** -0.5)
    w = torch.softmax(s.float(), dim=-1).to(q.dtype)
    return torch.einsum("bhgt,bhtd->bhgd", w, V)

def timeit(fn, iters=ITERS):
    for _ in range(5): fn()
    if DEV == "cuda": torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn()
        if DEV == "cuda": torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2]          # median, not mean
```

Time the recurrent step once (it does not depend on `T`), then sweep `T_GRID` allocating
`K, V` of shape `[BATCH, N_KV, T, D_H]` and a query of shape `[BATCH, N_KV, G, D_H]`.
Report, per row: KV MiB, attention ms, achieved GB/s, and the ratio `attn_ms / rec_ms`.

**Footprint check before you run.** At `T = 131072`: `K` and `V` are `1 × 8 × 131072 × 128 ×
2 B` = 256 MiB each. Trivially inside the `[M]` ≥62 GiB fast tier and nowhere near the 31 GiB
per-tensor hazard. Raising `BATCH` to 64 puts each tensor at 16 GiB and the footprint at
32 GiB — still legal, but one more doubling is not. Redo that arithmetic before you touch
`BATCH`, and remember the failure mode is a **silent hang at 0 CPU**, not an exception.

**Deliverables — three numbers, one plot, one honest caveat.**

1. **The bytes crossover.** Closed form, must print exactly **128**. If it does not, your
   `2` for K-and-V is missing or doubled.
2. **The wall-clock crossover** — the `T` at which `attn_ms / rec_ms` passes 1.0. `[M]` I got
   it **between 512 and 2048 in both of two runs**: run 1 gave 0.79 → 1.48, run 2 gave 0.85 →
   1.16. So the wall-clock crossover is roughly **8× later than the bytes crossover**, and
   the bracket is what reproduces.
3. **The gap at `T = 131072`.** Bytes ratio 1024×. `[M]` Measured time ratio **18.6× (run 1)
   / 28.4× (run 2)**. Explain the shortfall before you read on.

**The caveat, and it is the finding.** `[M]` The absolute timings are **not** repeatable to
better than ~2× between runs on this machine — the recurrent step measured 0.514 ms and then
0.281 ms; attention at `T=131072` measured 9.55 ms then 7.99 ms (medians of 30, same script,
same session-scoped env, minutes apart). Only the *bracket* of the crossover reproduced.
Report the bracket as `[M]` and the absolute times as run-specific; do not put a single-run
millisecond figure into `ASSUMPTIONS.md`.

**Where the missing 1024× went.** The recurrent step moves 512 KiB and takes ~0.3–0.5 ms,
i.e. ~1–2 GB/s against the `[M]` 199.9 GB/s fast tier — 99% of that time is Python dispatch
and kernel launch for five separate eager ops, not memory traffic. Attention at `T=131072`
reaches only 56–67 GB/s, about a third of the copy bandwidth. **So the byte-ratio argument
is an argument about a kernel you have not written yet.** A fused recurrent kernel is not a
performance nicety here; it is the difference between a 1024× advantage and an 18× one. That
is the same lesson as `repeat_kv` in the previous module, arriving from the opposite
direction.

---

### Exercise C — price the prefill, and find the chunk length nobody optimizes for

**Goal:** demonstrate that "constant memory" is a decode-time property, derive the
memory-optimal chunk length, and check it against what Mamba ships.

**Hardware:** one gfx1151 GPU; CPU fallback uses storage accounting instead of
`max_memory_allocated`, which is the more honest measurement anyway. **Runtime:** under a
minute. **Difficulty:** 3/5 — the derivation is the work.

First, on paper. Open `architecture/mamba/mamba_ssm/modules/ssd_minimal.py:34` and write down
the shapes of the two big allocations (lines 54 and 60). Derive:

```
prefill_bytes(cl)  =  4·B·H·L·( P·N/cl + cl )        →        cl* = √(P·N)
```

At Mamba-2's defaults (`mamba2.py:41,45`) that is `√(128·64) = 90.5`, against a shipped
`chunk_size` of 256 (`mamba2.py:59`).

Then measure. Allocate `A [B,H,C,cl]`, `X [B,C,cl,H,P]`, `Bt [B,C,cl,H,N]`, run `segsum`
(copy the shape logic from `ssd_minimal.py:23`) and the `states` einsum
(`"bclhn,bhcl,bclhp->bchpn"`, `ssd_minimal.py:60`), and record
`torch.cuda.max_memory_allocated()` around it. Sweep `cl ∈ {16, 32, 64, 128, 256, 512}` at
`B=1, H=32, L=8192, P=64, N=128`, fp32.

**Deliverables — three numbers.**

1. **The argmin.** `[M]` I measured the flat minimum at `cl ∈ {64, 128}`, both **514.0 MiB**,
   bracketing `cl* = 90.5`; 882.0 MiB at `cl=16` and 1536.5 MiB at `cl=512`. Your argmin must
   land in the same bracket.
2. **The gap between closed form and measurement.** `[M]` Measured exceeds closed form by
   **322–354 MiB** for `cl ≤ 128` (einsum intermediates) and by 1008 MiB at `cl=512`. Name
   the tensors responsible. The closed form is a lower bound; say so whenever you quote it.
3. **The extrapolation, and the fault locus.** Closed form only, nothing allocated: at
   `L = 1M`, `cl = 256`, the `states` term alone is **4.00 GiB** against a **1 MiB** decode
   state — 4096×. Then solve `states = 31 GiB` for `L/cl` and report the `(L, chunk_size)`
   pairs that land on `ASSUMPTIONS.md → large-tensor-fault-32gib`. `[M]` I get `L/cl = 31,744`
   chunks, i.e. `L ≈ 1.0M` at `cl = 32`.

**What a falsification would mean.** If the measured argmin is *not* near `√(P·N)`, either
the einsum is materializing something the closed form ignores (likely — check the constant
offset first) or the allocator is caching across iterations (call `empty_cache()` and
`reset_peak_memory_stats()` between rows). Either way, write it up: an allocator artifact
that shifts a memory optimum is exactly the sort of thing that silently invalidates a
chunk-size ablation.

**Do not skip the caveat.** The fused Triton path does not materialize the segsum matrix
(`ssd_combined.py:375` computes chunk states directly), so half of your U-curve is a property
of the reference implementation and not of production Mamba. Quoting the total as "Mamba's
prefill memory" would be wrong. Quoting the `states` term is correct.

---

## 7. Self-check

Answers at the end of the file. Do not scroll.

1. A colleague proposes replacing a 48-layer model's KV cache with a constant recurrent
   state and claims "1000× less memory." At what context length is the claim exactly
   *reversed* — the state larger than the cache — and what does the answer depend on?

2. Exercise A measures a *smooth* degradation: delta-rule accuracy 1.000 → 0.901 → 0.526 as
   `n/d_k` goes 0.5 → 1.0 → 2.0. The literature reports *cliffs*: at 1.3B params and 4K
   training length, Mamba-2 scores 47.6 on the S-NIAH-3 word-in-haystack needle at 2K
   context and **4.6** at 4K `[C]` 2412.06464. Give two mechanisms by which a smooth
   per-association degradation becomes a cliff in a benchmark number, and say which one you
   could test on the Z13 in an afternoon.

3. In Exercise A, pure accumulation beats the delta rule on write-once random keys, and loses
   badly to it when keys are rewritten. State the property of `I − β k kᵀ` that produces both
   results, in one sentence.

4. The read norm `‖o‖` has AUROC 0.463 as a predictor of read correctness when pooled across
   `n`, and 0.58–0.63 within a fixed `n`. The pooled number is below 0.5. What has happened,
   what is the name for it, and what is the one-line rule it implies for any memory-health
   metric you ship?

5. You are asked whether Proteus should adopt a Gated DeltaNet layer to "get selective
   forgetting." Answer using the code, not the paper.

6. At 128k context on our machine, replacing Laguna-S's entire KV cache with a 24 MiB
   constant state improves decode from 8.5 tok/s to 11.8 tok/s — 1.38×, against a 1024× byte
   reduction. Where did the other three orders of magnitude go, and what *is* the argument
   for constant state on this hardware?

---

## 8. What is still unsolved here

Everything here is testable at 20M–300M params on one gfx1151 GPU with a `[M]` ≥62 GiB fast
tier, single-device only, individual tensors under 31 GiB. Every item needs a pre-registered
hypothesis card before it runs.

1. **Does the `n < d_k` capacity behaviour survive training?** Exercise A shows the algebra
   degrades smoothly with random keys. Zoology's `d ≥ N` result `[C]` 2312.04927 and the
   published NIAH tables show something sharper in trained models. Those are compatible —
   training changes the key geometry and the decision rule is exact match — but nobody has
   measured the untrained algebraic curve and the trained task curve on the *same* shapes.
   That is a one-week experiment and it is the natural calibration of the rig against a
   published shape.

2. **Attribution: is the cliff caused by decay or by interference?** Ablate the gate
   (`α = 1`) and the delta term (`β` fixed vs learned) *separately*. §4.1's three-field config
   split exists for this. Exercise A gives the untrained prediction: the gate should hurt on
   rewrite-heavy data and help on stale-data-heavy data, and the delta term should carry all
   content-dependent selectivity.

3. **Holding parameters fixed, does state size alone move the recall cliff?** Sweep
   `d_k × d_v` in a Gated DeltaNet at 20M–300M. This is the matched-state-versus-matched-params
   dispute `[C]` 2605.22791 run as an ablation rather than argued. Most comparisons match
   parameter count; these give different rankings and papers rarely report both.

4. **Can a miss signal be synthesized at all?** `[M]` The read norm cannot do it. A learned
   probe on the full state, or on read-margin statistics, might. This is the most
   Mnemosyne-shaped question in the module: if a usable confidence signal exists, every
   fallback pattern you have ever built becomes applicable, and if it provably does not, that
   is a stronger negative result than the field currently has.

5. **Does a bf16 recurrent state change the recall cliff?** `[M]` Exercise A says no at the
   algebraic level — bf16 and fp64 states give identical retrieval accuracy to three decimals.
   But Mamba's chunk states are fp32 *by construction* (`ssd_combined.py:375`), which suggests
   somebody found a reason. Given `bf16-numerics-unproven` is open on this hardware, run a
   trained model's state in bf16 vs fp32 and measure **recall, not loss**. This ties the
   Hardware Validation Gate to a research result rather than treating it as maintenance.

6. **Does the prefill state materialization hit our 32 GiB single-tensor fault?** `[A]` The
   arithmetic in §3.9 says a 1M-token prefill at `chunk_size = 32` lands on it. Nobody has run
   it. The failure mode is a **silent hang at 0 CPU**, so a long run would stall rather than
   crash — which makes this a reliability question, not just a capacity one.

7. **What is the smallest exact cache that recovers full-attention recall?** Sweep a
   HOLA-style bounded cache `[C]` 2607.02303 against recall at fixed state size. The output is
   a capacity-planning curve denominated in bytes, directly usable against the `[M]` 62 GiB
   budget, and it is the question the whole field has drifted toward without stating.

8. **Does 3:1 hold at our scale?** `[C]` 2507.06457 trained 72 models across 6 linear variants
   × 5 ratios and recommends 3:1–6:1, finding recall improves sharply as you go *below* 3:1.
   Its smallest arm is 340M, just above our ceiling. If the optimal ratio drifts with scale,
   every small-scale hybrid ablation in the literature is mis-calibrated, including ours.

9. **The eager-kernel gap.** `[M]` Exercise B measures 18–28× instead of the 1024× the bytes
   predict, entirely because a five-op eager recurrent step costs more in launch overhead than
   its 512 KiB costs in bandwidth. Nobody reports realized-versus-theoretical decode advantage
   for constant-state models; every published throughput number is from a fused kernel, and
   the *gap* is unmeasured. That is the same attribution failure this lab exists to attack.

**Contested — do not let this module imply consensus.**

- **Does hybrid linear attention work at production scale?** MiniMax shipped M2 (229.9B total
  / 9.8B active `[C]` 2605.26494) as **full attention with plain MHA**, publishing a note that
  they found no efficient-attention variant reliably matching full attention across reasoning,
  coding and agentic tasks — hybrids matched on MMLU/BBH/MATH/LongBench but showed clear
  deficits on multi-hop reasoning at scale, across hundreds of billions to trillions of
  continued-pretraining tokens. Kimi Linear `[C]` 2510.26692 claims the opposite under
  matched-scale pretraining (3:1 KDA:MLA, 75% KV reduction, 6× decode throughput at 1M
  context). **Both are shipping-product retrospectives with commercial incentives; neither is
  a controlled ablation.** Treat the disagreement as the open question of the track.
- **Does the hybrid ratio set a capability ceiling or only a training-speed knob?**
  `[C]` 2507.06457 finds recall improves sharply as full-attention layers increase,
  particularly below 3:1; `[C]` 2606.15378 argues different configurations converge given
  enough training and that the efficient-attention choice governs *how fast* long-context
  ability emerges, not its limit. Same year, incompatible framings. The resolution likely
  depends on token budget — which is exactly the axis a small rig can attack.
- **Is state-tracking expressivity the same thing as recall?** It is not (§3.6), and the
  literature routinely presents both under "expressivity."
- **Do MQAR results transfer?** Zoology attributes 82% of a real perplexity gap to associative
  recall, the strongest transfer evidence available. But MQAR is synthetic and no source
  demonstrates that an MQAR capacity curve predicts a downstream task curve *quantitatively*.
- **A note on evidence quality.** None of the recall tables in `research/memory/constant-state-memory.md`
  report seed counts or confidence intervals. By this lab's own standard they are anecdotes
  with good provenance, not measurements. The same standard applies to Exercise B's absolute
  timings, which is why only the bracket is tagged `[M]`.

**Where this module sharpens the survey note rather than contradicting it.** The note's §3 is
titled "The recall failure, concretely" and presents the failure as a cliff. Exercise A shows
the *algebra* has no cliff — it has a smooth `1/√(1 + (n−1)/d_k)` decay that matches
prediction to 0.01. That is not a disagreement about facts; the note itself flags the one
table with a visible cliff as a broken harness. It is a sharpening of the mechanism: **the
smoothness is in the state, and the cliff is manufactured by the decision rule and by
training.** Self-check 2 works through why. Everything else in the note is confirmed by
measurement, including the code-derived claim about the gate, which Exercise A report 3 turns
into a number.

---

## Answers to the self-check

**1.** At `T* = (H·d_k·d_v) / (2·n_kv·d_h)`. For the two shapes in §3.7 that is
`(16·128·128)/(2·8·128) = 128 tokens`. Below 128 tokens of context the constant state is
strictly *larger* than the cache it replaces. The answer depends only on the two **shape**
ratios — the dtype cancels, and neither parameter count nor layer count appears (both scale
the two sides equally). Practical consequence: any benchmark of a constant-state model at
chat-turn context lengths is measuring a regime where the technique loses, and any vendor
number quoted without a context length is unreadable.

**2.** Two mechanisms. **(a) Exact match compounds.** If a benchmark scores a multi-token
answer as all-or-nothing, per-token accuracy `p` over `m` tokens becomes `p^m`. A smooth slide
from `p = 0.95` to `p = 0.80` becomes 0.77 → 0.33 at `m = 5` — a cliff, manufactured entirely
by the scoring rule. **(b) Training changes the key geometry.** Below capacity a model can
learn near-orthogonal keys and recall is near-perfect; past capacity, no key assignment exists
that keeps interference below the read's discrimination threshold (§3.5's Welch bound), and
the learned solution collapses to something qualitatively different — typically recency,
which the recency profile in Exercise A already shows the algebra doing on its own.
**Testable in an afternoon:** (a). Re-score your Exercise A output as "all `n` reads correct"
instead of "fraction correct" and watch the same numbers become a cliff, with no change to
the model. (b) needs training runs.

**3.** `I − β k kᵀ` at `β = 1` is the projection that annihilates the `k` direction and leaves
everything orthogonal to `k` untouched — so it destroys exactly as much prior content as the
new key overlaps with the old ones, which is pure cost when each key is written once and
exactly the right behaviour when a key is being rewritten.

**4.** `‖o‖` rises with `n` (more terms in the superposition) while accuracy *falls* with `n`,
so pooling across `n` mixes a within-group positive signal with a between-group negative one
and the aggregate inverts. This is **Simpson's paradox**, and the confounder is load. The
rule: **never aggregate a memory-health metric across a variable that also drives the
outcome** — stratify by context length, or report the metric per bucket, or you will ship a
dashboard whose green means red. This is the cheapest available warning that the observability
patterns you own do not transfer to this substrate unmodified.

**5.** "Selective forgetting" is not what the gate does. Read
`architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:54`: the decay is one
scalar per head multiplying the *entire* `d_k × d_v` matrix, so every stored association is
attenuated identically every step. There is no per-key or per-channel TTL, and this is not a
kernel limitation — the production kernel implements per-key (`USE_GK`) and per-channel
(`USE_GV`) gates at
`architecture/flash-linear-attention/fla/ops/gated_delta_rule/fused_recurrent.py:145` and the
layer does not pass them. All content-dependent selectivity lives in the **delta** term,
which erases exactly one direction via the read-before-write at
`architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:56`. So: adopt it for the *delta* term, and be explicit that you are
also inheriting an indiscriminate global decay that `[M]` Exercise A measures as actively
harmful on rewrite-heavy data at fixed `α` (0.583 vs 0.875 at `n=64`). If you want genuine
per-channel selectivity, that is Gated DeltaNet-2 `[C]` 2605.22791 or RWKV-7 `[C]` 2503.14456,
both of which exist because of precisely this gap.

**6.** Amdahl. The KV read is only one term of decode; the weight read is the other, and on
Laguna-S it is 17.0 GB = 85.0 ms per token at our `[M]` 199.9 GB/s, which the state change
does not touch. Removing 32.6 ms of cache traffic from a 117.6 ms step is 1.38×, and no byte
reduction can ever beat 117.6/85.0 = 1.38× at that context. **The real argument is about
batch, not latency.** Weight traffic is *shared* across a batch and cache traffic is *private*
per sequence, so the weight term's arithmetic intensity is the batch size while the attention
term's is invariant (previous module, §3.9). Our roofline ridge is `[M]` ≈105 FLOP/byte, so
you need batch ≈105 to stop being bandwidth-bound — and at a plausible Proteus shape a KV
cache runs out of capacity at batch 40 by 32k context and batch 10 by 128k, i.e. it becomes
*structurally impossible* to reach the ridge beyond ≈12,800 tokens. A 1.5 MiB constant state
holds 41,943 sequences at any context. Constant state does not make one sequence fast; it
makes the machine reachable.

---

## Sources

**Local measurements (`[M]`)**, all 2026-07-26, lab venv `C:\venvs\lab`, torch
`2.12.0a0+rocm7.13.0a20260313` (HIP 7.2.0), native Windows, gfx1151, session env from
`scripts/activate-lab.ps1` (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` **not** set):

- **Exercise A** — `d_k = d_v = 64`, seeds {0,1,2}, CPU, fp64 state unless stated. Fully
  deterministic; reruns are bit-identical. Capacity curve, orthogonal-key control, interleaved
  overwrite test, dtype sweep, recency profile, and the `‖o‖` AUROC. Closed-form check of
  `cos = 1/√(1+(n−1)/d_k)` agrees to ≤0.01 absolute over `n ∈ [8, 256]`.
- **Exercise B** — gfx1151, bf16, `H=16, d_k=d_v=128`, `n_kv=8, d_h=128, G=6`, batch 1,
  median of 30 iterations, `T ∈ {128 … 131072}`. **Two independent runs.** The wall-clock
  crossover bracket (512 < `T*` < 2048) reproduced; absolute timings varied up to ~1.8×
  between runs and are reported as run-specific, not as `[M]` constants.
- **Exercise C** — gfx1151, fp32, `B=1, H=32, L=8192, P=64, N=128`, `cl ∈ {16 … 512}`,
  `torch.cuda.max_memory_allocated` with `empty_cache()` + `reset_peak_memory_stats()` per row.
  Single run; the argmin bracket is the transferable result.

**Repo (`[M]` / register rows)**

- `ASSUMPTIONS.md` — `gpu-fast-tier-size` (≥62 GiB at ~200 GB/s, single run per arm),
  `large-tensor-fault-32gib`, `kv-per-token-laguna` (192 KiB/token exact),
  `reference-model` (48 layers, 12 full + 36 sliding, GSSS, `w`=512, `n_kv`=8, `d_h`=128),
  `bf16-numerics-unproven`, `single-device-only`, `gemm-throughput-below-reference`
  (20.9 TFLOP/s bf16 at 8192³, giving the ≈105 FLOP/byte ridge), `torch-build`.
- `notebook/uma-carveout-controls-fast-tier.md` — the 199.9 GB/s figure at a 62 GiB footprint.
- `research/memory/constant-state-memory.md` — the survey this module teaches. No number here
  contradicts it; §8 states the one place this module sharpens it.
- `research/synthesis.md` — the decision to build the instrument rather than another policy.
- `curriculum/attention-variants-and-kv-cost.md` — the KV product, the `AI = 2G/b` derivation,
  and the ridge point this module reuses.
- Code pointers: every `file:line` in §5 was opened and the named symbol confirmed on the
  named line on 2026-07-26, against the revisions in `research/reference/PROVENANCE.md`.

**arXiv (`[C]`)**

- `1706.03762` — *Attention Is All You Need* (2017). The exact log this family replaces.
- `2312.00752` — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (Dec 2023).
  Input-dependent selectivity plus the hardware-aware scan.
- `2312.04927` — *Zoology: Measuring and Improving Recall in Efficient Language Models*
  (Dec 2023). MQAR, the `d ≥ N` requirement, the 82%-of-the-gap attribution.
- `2401.04658` — *Lightning Attention-2* (Jan 2024). A kernel, not an architecture.
- `2402.18668` — *Simple linear attention language models balance the recall-throughput
  tradeoff* (Based, Feb 2024, rev. Mar 2025). State size as a dial.
- `2405.04517` — *xLSTM: Extended Long Short-Term Memory* (May 2024). mLSTM's normalizer state.
- `2405.21060` — *Transformers are SSMs* (Mamba-2 / SSD, May 2024). The chunked scan.
- `2406.06484` — *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*
  (DeltaNet, Jun 2024).
- `2411.12537` — *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*
  (Nov 2024, rev. Mar 2025). Why `β ∈ (0,2)` matters, and why it is not recall.
- `2412.06464` — *Gated Delta Networks: Improving Mamba2 with Delta Rule* (Dec 2024). The
  S-NIAH tables quoted in the survey note.
- `2501.08313` — *MiniMax-01: Scaling Foundation Models with Lightning Attention* (Jan 2025).
- `2503.14456` — *RWKV-7 "Goose" with Expressive Dynamic State Evolution* (Mar 2025).
- `2506.04761` — *Log-Linear Attention* (Jun 2025, ICLR 2026). Abandons O(1) deliberately.
- `2507.06457` — *A Systematic Analysis of Hybrid Linear Attention* (Jul 2025, rev. Jun 2026).
  72 trained models; 3:1–6:1.
- `2507.16577` — *Scaling Linear Attention with Sparse State Expansion* (Jul 2025, rev. Oct 2025).
- `2510.26692` — *Kimi Linear: An Expressive, Efficient Attention Architecture* (Oct 2025).
- `2603.15569` — *Mamba-3: Improved Sequence Modeling using State Space Principles* (Mar 2026).
  Complex-valued state; downstream average and retrieval capacity ranking differently.
- `2605.05066` — *The Impossibility Triangle of Long-Context Modeling* (May 2026).
  *Single-author preprint, unreplicated.*
- `2605.11196` — *Variational Linear Attention* (May 2026). The MQAR table whose softmax
  control Exercise A reproduces as an artifact. *Two authors, one seed.*
- `2605.22791` — *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention* (May 2026).
  The exact per-layer state figure; the matched-state argument.
- `2605.26494` — *The MiniMax-M2 Series* (May 2026). The full-attention counter-position.
- `2606.15378` — *Rethinking the Role of Efficient Attention in Hybrid Architectures* (Jun 2026).
- `2607.02303` — *A Hippocampus for Linear Attention (HOLA)* (Jul 2026). Bounded exact cache
  with a residual-magnitude admission policy. *Single author, unreplicated.*
- `2607.07386` — *Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity*
  (Jul 2026). State size and per-token compute as separable knobs.
- `2607.07953` — *Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer
  Routing* (Jul 2026). The unified-notation comparison.
- `2607.17419` — *Kernelized Linear Attention: Breaking the Capacity Wall with Symmetric Cones*
  (KATA, Jul 2026). The Welch interference floor as the capacity characterization.

**Non-arXiv**

- MiniMax, "Why Did M2 End Up as a Full Attention Model?" —
  https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model
- LMSYS, "No Free Lunch: Deconstruct Efficient Attention with MiniMax M2" —
  https://www.lmsys.org/blog/2025-11-04-miminmax-m2/ (Nov 2025)
