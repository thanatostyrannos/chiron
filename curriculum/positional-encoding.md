---
title: Positional encoding — why attention needs position, and how RoPE, YaRN, NoPE and partial rotary actually work
version: 1.0.0
date: 2026-07-26
track: B — Modern architecture
prereqs: transformer-forward-pass, attention-variants
worked_example: poolside Laguna S 2.1, read from the committed artifact
---

# Positional encoding

## What this module settles

Self-attention is a **set** operation — it computes the same answer for a sentence and
for the same words shuffled — so position has to be injected deliberately, and every
scheme for doing it is a different answer to one question: *should the attention logit
depend on absolute position, relative distance, or neither?* RoPE answers "relative"
by rotating queries and keys by an angle proportional to their absolute position, which
makes the logit a function of the difference of positions through a three-line identity
you can verify by hand — and that identity is exactly what YaRN, partial rotary and NoPE
each partially give up in exchange for length generalization. By the end you will be
able to read Laguna's `rope_parameters` block, reproduce all four of its published
constants from first principles, and say precisely why a KV eviction policy that
renumbers positions is either correct or silently corrupting depending on a design
decision nobody writes down.

---

## 1. Theory in plain language

### 1.1 The defect being repaired

Take the attention output for query position `i`:

```
out_i  =  Σ_j  softmax_j( q_i · k_j / √d ) · v_j
```

Read the right-hand side as a systems person would read a query plan. It is a
**reduction over a set**. The index `j` appears only as a subscript for *fetching*
`k_j` and `v_j`; it never appears as a *value* in any arithmetic. So if you permute the
input tokens, every `k_j` and `v_j` moves with its token, the sum runs over the same
multiset, and the output permutes identically. The layer cannot tell "the cat sat on the
mat" from "mat the on sat cat the".

This is not a bug in the implementation. It is the defining property of attention, and
it is the reason attention parallelizes at all — an RNN gets order for free precisely
because it cannot be parallelized over the time axis. Attention traded order for
parallelism and then had to buy order back.

> **Systems bridge.** This is the difference between a `SET` and a `LIST` in your data
> model, or between an unordered Kafka topic and a partitioned one. You did not get
> ordering for free; someone had to put a sequence number in the record.
>
> **Where it breaks:** in a distributed log, the sequence number is a separate field you
> can read, rewrite, and validate. In a transformer, position gets *mixed into the
> payload*, and after mixing there is no field to read back. That single difference is
> responsible for most of the hard parts of this module and for one entire class of KV
> cache bugs.

### 1.2 The causal mask already leaks some of it

One qualification, because it matters later. Decoder attention is causally masked:
query `i` may only attend to keys `0…i`. That means the *sets* are nested — position 3's
view is a subset of position 4's view — and nested sets carry a total order. A one-layer
causal transformer only sees a multiset per position, but a two-layer one can compare
"the summary of my prefix" against "the summary of your prefix" and recover ordering.

So a decoder with **no positional encoding at all** is not position-blind. It is
position-*implicit*. That is the NoPE result, and it is why NoPE is a real option rather
than a thought experiment `[C]` (arXiv [2305.19466](https://arxiv.org/abs/2305.19466),
2023).

### 1.3 The four families, and what each one broke

| Family | Mechanism | What it bought | What it cost |
|---|---|---|---|
| **Absolute, learned** (GPT-2, nanoGPT) | a table `p_m ∈ R^d`, one row per position, **added to the token embedding** before layer 0 | trivial; one `nn.Embedding` | hard cap at `block_size` rows; rows past training length are untrained noise, not empty space; the logit depends on absolute `m` and `n`, not on `m−n` |
| **Absolute, sinusoidal** `[C]` ([1706.03762](https://arxiv.org/abs/1706.03762), 2017) | fixed `sin`/`cos` of geometric frequencies, still **added** | no table, defined at any `m` | still additive, so the logit still mixes absolute and content terms (§2.2) |
| **Relative bias** `[C]` ([1803.02155](https://arxiv.org/abs/1803.02155), 2018; T5-style buckets; ALiBi `[C]` [2108.12409](https://arxiv.org/abs/2108.12409), 2021) | add a term that depends on `m−n` **to the logit** | genuinely relative by construction | the term is a function of the *pair*, so it cannot be folded into a cached key; it must be recomputed against the whole cache every step |
| **Rotary (RoPE)** `[C]` ([2104.09864](https://arxiv.org/abs/2104.09864), 2021) | **rotate** `q` and `k` by an angle proportional to their own absolute position | relative logit *and* a per-token transform that can be baked into the cache | the position is now baked into the cached bits and cannot be changed, read back, or checksummed |

That last row is why RoPE won, and the phrasing matters. RoPE is the only scheme in the
list that is **relative in effect but absolute in implementation**. Every token is
transformed using only its own index, so the transform is a per-token map that commutes
with caching. A relative bias is a per-*pair* map, and per-pair maps do not cache.

For a storage engineer the trade reads like this: relative bias is a join predicate
evaluated at query time; RoPE is a computed column materialized at write time. RoPE won
for the same reason materialized columns usually win — until you need to change the
column, at which point you discover you have to rewrite every row.

---

## 2. The math that actually matters

Notation, all of it, once:

| Symbol | In words |
|---|---|
| `d` | head dimension — the width of one attention head's query/key vector. Laguna: `head_dim = 128` |
| `d_rot` | how many of those `d` channels actually get rotated. Laguna full-attention layers: **64**; sliding layers: **128** |
| `m`, `n` | absolute positions (integers) of the query token and the key token |
| `δ = n − m` | relative distance from query to key. Negative under causal masking (keys are in the past) |
| `q`, `k` | the query and key vectors for one head, *before* rotation |
| `θ` | the **RoPE base**, a scalar hyperparameter. Laguna full: `500000`; sliding: `10000` |
| `i` | index of a **channel pair**, running `0 … d_rot/2 − 1`. Laguna full: 32 pairs |
| `ω_i` | angular frequency of pair `i`, in radians per token |
| `λ_i` | wavelength of pair `i`, in tokens per full rotation |
| `s` | YaRN scale factor (`factor` in the config). Laguna full: **128** |
| `L_orig` | context length the model was actually pretrained at. Laguna: **8192** |
| `t` | YaRN's attention temperature term (`attention_factor`). Laguna full: **1.4852030263919618** |

### 2.1 Why *adding* a position vector fails to give a relative logit

Suppose you add `p_m` to the token embedding: `x_m = e_m + p_m`. The query and key are
linear projections, so `q_m = W_q(e_m + p_m)` and `k_n = W_k(e_n + p_n)`. The logit
expands into four terms:

```
q_m · k_n  =  (W_q e_m) · (W_k e_n)      content  x  content
           +  (W_q e_m) · (W_k p_n)      content  x  position(n)
           +  (W_q p_m) · (W_k e_n)      position(m) x content
           +  (W_q p_m) · (W_k p_n)      position(m) x position(n)
```

Only the fourth term has any chance of collapsing to a function of `n − m`, and only if
`W_q^T W_k` happens to have the right structure — nothing forces it. The two cross terms
are irreducibly functions of **absolute** `m` and `n`. This is the precise reason "the
sinusoidal encoding is a linear function of relative offset" is a true statement about
`p` that does not translate into a true statement about the logit.

The practical consequence you can see in code: a learned absolute table has exactly
`block_size` rows and the forward pass asserts on it. There is no fault-in path for
position `block_size + 1`, because the row was never trained — allocating it gives you
noise, not zeroes. See `training/nanogpt/model.py:173` and `:178`.

> **Systems bridge.** A learned position table is a statically sized array indexed by
> offset, with a bounds check. You know this shape.
>
> **Where it breaks:** you can grow an array. You cannot grow this one. Rows beyond the
> trained range are not *unallocated*, they are *allocated and garbage*, and the failure
> is semantic rather than a segfault. There is no `ENOSPC`; the model just answers wrong.

### 2.2 The functional equation RoPE solves

State the goal as a constraint before looking at the answer. We want per-token maps
`f_q(q, m)` and `f_k(k, n)` such that

```
⟨ f_q(q, m) , f_k(k, n) ⟩  =  g(q, k, n − m)
```

for **some** function `g`. In words: *"transform the query using only its own position,
transform the key using only its own position, and require that the resulting inner
product depend on the two positions only through their difference."* Per-token maps are
what make caching possible; difference-only dependence is what makes it relative. The
question is whether both can hold at once.

### 2.3 The two-dimensional answer, proved in three lines

Take `d = 2`. Let

```
R(φ)  =  [ cos φ   −sin φ ]
         [ sin φ    cos φ ]
```

the standard 2-D rotation by angle `φ`. Define `f_q(q, m) = R(mω) q` and
`f_k(k, n) = R(nω) k` for a fixed frequency `ω`. Then

```
⟨R(mω) q , R(nω) k⟩ = (R(mω) q)ᵀ (R(nω) k) = qᵀ R(mω)ᵀ R(nω) k
```

Rotation matrices are orthogonal, so `R(φ)ᵀ = R(−φ)`, and rotations compose additively,
`R(a)R(b) = R(a+b)`. Therefore

```
qᵀ R(−mω) R(nω) k  =  qᵀ R((n − m) ω) k
```

which is a function of `n − m` only. That is the whole theorem.

Do not take "rotations compose" on faith — multiply it out once:

```
R(φ)ᵀ R(ψ) = [  cos φ   sin φ ] [ cos ψ   −sin ψ ]
             [ −sin φ   cos φ ] [ sin ψ    cos ψ ]

  top-left      =  cos φ cos ψ + sin φ sin ψ  =  cos(ψ − φ)
  top-right     = −cos φ sin ψ + sin φ cos ψ  = −sin(ψ − φ)
  bottom-left   = −sin φ cos ψ + cos φ sin ψ  =  sin(ψ − φ)
  bottom-right  =  sin φ sin ψ + cos φ cos ψ  =  cos(ψ − φ)
```

which is `R(ψ − φ)`. Substituting `φ = mω`, `ψ = nω` gives `R((n−m)ω)`. Note what did
*not* happen: no cross terms, no content–position mixing. Rotation is
content-preserving in a way addition is not, because a rotation acts on the *whole*
vector rather than being summed into it.

### 2.4 Scaling to `d` dimensions: a bank of dials

Split the `d_rot` rotated channels into `d_rot/2` independent planes and give plane `i`
its own frequency:

```
ω_i  =  θ^(−2i / d_rot)          radians per token,   i = 0 … d_rot/2 − 1
```

and build the block-diagonal matrix `R_m = diag( R(m ω_0), R(m ω_1), … )`. Block-diagonal
orthogonal matrices compose block by block, so §2.3 holds in every plane independently
and the total logit is the sum:

```
⟨R_m q, R_n k⟩ = Σ_i  (q_2i k_2i + q_2i+1 k_2i+1) · cos(δ ω_i)
                    + (q_2i+1 k_2i − q_2i k_2i+1) · sin(δ ω_i)
```

with `δ = n − m`. In words: **for each channel pair, the dot product of the two
2-vectors is weighted by a cosine of the distance, and their cross product is weighted by
a sine of the distance.**

There is a cleaner reading. Write pair `i` of the query as a complex number
`z_q = q_2i + j·q_2i+1` and likewise `z_k`. Then plane `i` contributes

```
Re( conj(z_q) · z_k · e^{ j δ ω_i } )  =  |z_q| |z_k| · cos( δ ω_i + ∠z_k − ∠z_q )
```

**The attention logit is a Fourier series in the relative distance `δ`.** The frequencies
`ω_i` are fixed by `θ` and never learned. The amplitudes `|z_q||z_k|` and the phase
offsets `∠z_k − ∠z_q` *are* learned, per query–key pair. A head does not learn "attend 5
tokens back"; it learns a set of Fourier coefficients whose sum happens to peak 5 tokens
back.

The useful unit is not frequency but **wavelength**:

```
λ_i  =  2π / ω_i  =  2π · θ^(2i / d_rot)      tokens per full rotation
```

> **Systems bridge.** A mixed-radix odometer, or a bank of clocks with geometrically
> spaced periods. Position is not stored as an integer anywhere; it is encoded as
> *phase* across `d_rot/2` dials. Fast dials distinguish "previous token" from "two
> tokens ago"; slow dials distinguish "same paragraph" from "same file". Reading a
> distance means reading the whole bank at once, which makes the code distributed and
> redundant in the way an erasure code is.
>
> **Where it breaks, and this is the load-bearing break for Mnemosyne.** An odometer can
> be re-read at any time. A KV cache stores **post-rotation** keys — `apply_rotary_pos_emb`
> runs before the cache write — so each cached key has its absolute position baked in as
> phase, entangled with its content, unrecoverable. That is the equivalent of writing a
> *physical* address into a cache line instead of a virtual one: no remapping, no
> re-packing, no renumbering without recomputation. And unlike a bad physical address,
> there is no fault, no parity error, no counter. The model just answers wrong.

### 2.5 The implementation you will actually read

HuggingFace does not pair channel `2i` with `2i+1`. It pairs channel `j` with channel
`j + d_rot/2`, which is the same rotation under a permutation of channels. The code
(`models/laguna-s/modeling_laguna.py:263` and `:300`) is:

```python
def rotate_half(x):                      # (a, b) -> (-b, a) on the two halves
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
```

with `cos` and `sin` built as `torch.cat((freqs, freqs), dim=-1)` so that entry `j` and
entry `j + d_rot/2` carry the same angle. Expand it for the pair `(j, j+h)`, `h = d_rot/2`:

```
q'_j     = q_j     · cos(m ω_j)  −  q_{j+h} · sin(m ω_j)
q'_{j+h} = q_{j+h} · cos(m ω_j)  +  q_j     · sin(m ω_j)
```

which is exactly `R(m ω_j)` applied to `(q_j, q_{j+h})`. Same math, different channel
order.

**The trap:** the two conventions ("interleaved", GPT-J style; "split-half", GPT-NeoX
style) are *not* interchangeable at the weight level. Converting a checkpoint between
frameworks without permuting `q_proj` and `k_proj` rows produces a model that loads, runs,
emits plausible-looking tokens, and is quietly wrong. It is a byte-order bug with no
magic number to catch it.

### 2.6 Laguna's two dials banks, computed

`[M]` computed 2026-07-26 from `models/laguna-s/config.json:42-58`; cross-checked
against `research/memory/long-context-behavior.md` §2.

**Full-attention layers** — `θ = 500000`, `partial_rotary_factor = 0.5` so `d_rot = 64`,
giving **32 dials**:

| pair `i` | `λ_i` (tokens) | rotations inside the 8192 training window |
|---|---|---|
| 0 | 6.28 | 1304 |
| 8 | 167.1 | 49.0 |
| 9 | 251.8 | 32.5 |
| 17 | 6 695 | 1.22 |
| 18 | 10 086 | 0.81 |
| 31 | 2 084 765 | 0.0039 |

**Sliding-window layers** — `θ = 10000`, `partial_rotary_factor = 1.0` so `d_rot = 128`,
giving **64 dials**, longest wavelength **54 410** tokens.

Two things to read off this that are not in the model card.

**(a) The fastest dial swept its phase space 1304 times during pretraining. The slowest
swept 0.4% of one rotation — ever.** Asking the model for position 1,000,000 is not
"a longer input"; it is *phase values on the slow dials that were never observed*. That
is out-of-distribution input, not out-of-range input, and it is the entire reason
extending context is not a config change.

**(b) On the sliding layers, dial 31 has `λ = 544` — the first wavelength longer than the
512-token window.** So 33 of the 64 dials never complete a rotation inside the window
they operate in; within 512 tokens they act as a near-linear monotone distance ramp
(`sin(δω) ≈ δω` for small `δω`) rather than as a periodic code. Those layers are running,
in effect, a 31-dial rotary code plus an ALiBi-flavoured linear tail. Nobody documents
this; it falls out of the two numbers.

### 2.7 The three scalings, in order, with Laguna's arithmetic

**Position Interpolation (PI)** `[C]` ([2306.15595](https://arxiv.org/abs/2306.15595),
Jun 2023). Replace `m` with `m/s`; equivalently `ω_i → ω_i / s` on **every** dial.
Positions `0 … s·L_orig` now land inside the phase range that was observed in training.
The cost is uniform and brutal: the fast dials, which the model uses to distinguish
adjacent tokens, slow down by the same `s`. At `s = 128`, neighbouring tokens are
`1/128` of the rotation apart that they used to be. Local discrimination collapses.

**NTK-aware scaling.** Do not touch `ω` directly; change the base:
`θ → θ · s^(d_rot/(d_rot − 2))`. Because `ω_i = θ^(−2i/d_rot)`, a base change scales dial
`i` by roughly `s^(2i/d_rot)` — near-identity on the fast dials, near `1/s` on the slow
ones. Right shape, arrived at implicitly, inexact at both ends.

**YaRN** `[C]` ([2309.00071](https://arxiv.org/abs/2309.00071), 2023). Make the split
explicit and per-band, then add a temperature.

*(a) The per-band ramp.* For dial `i`, let `r_i = L_orig / λ_i` be the number of
rotations it completed inside the original training window. Then:

- `r_i ≥ β_fast` → **extrapolate**: leave `ω_i` alone.
- `r_i ≤ β_slow` → **interpolate**: `ω_i → ω_i / s`.
- in between → linear blend.

To implement it you need "which dial index has exactly `r` rotations", which is the
inverse of `r = L / λ(i)` with `λ(i) = 2π θ^{2i/d_rot}`:

```
r = L / (2π θ^{2i/d})      →   θ^{2i/d} = L / (2π r)
                           →   (2i/d) ln θ = ln( L / (2π r) )
                           →   i = d · ln( L / (2π r) ) / (2 ln θ)
```

That derivation is line-for-line `find_correction_dim` at
`architecture/transformers/src/transformers/modeling_rope_utils.py:425`. Note the result
is a **pair index**, which is why the ramp that consumes it runs over `dim // 2` entries.

Now put Laguna's numbers in. `d_rot = 64`, `θ = 500000`, `L_orig = 8192`, `β_fast = 32`,
`β_slow = 1`:

```
low   = 64 · ln( 8192 / (32 · 2π) ) / (2 · ln 500000)
      = 64 · ln(40.744) / 26.2447
      = 64 · 3.70736 / 26.2447   =  9.0406     → floor → 9

high  = 64 · ln( 8192 / (1 · 2π) ) / (2 · ln 500000)
      = 64 · ln(1303.80) / 26.2447
      = 64 · 7.17304 / 26.2447   = 17.4921     → ceil  → 18
```

`[M]` computed 2026-07-26; identical to the values in
`research/memory/long-context-behavior.md` §4. The blend itself
(`modeling_rope_utils.py:436`, `:454`, `:455`) is:

```
ramp(i)   = clamp( (i − 9) / (18 − 9), 0, 1 )
ω_i^new   = (ω_i / 128) · ramp(i)  +  ω_i · (1 − ramp(i))
```

So, counting dials: `ramp(9) = 0` exactly, and `ramp(18) = 1` exactly.

| dials | count | treatment |
|---|---|---|
| `i = 0…9` | **10** | untouched — pure extrapolation |
| `i = 10…17` | **8** | linearly blended |
| `i = 18…31` | **14** | fully divided by 128 |

> **Consistency note, folded back.** `research/memory/long-context-behavior.md` §4 states
> this split as 9 / 9 / 14. The difference is one dial: whether `i = 9`, where the ramp
> evaluates to exactly zero and the frequency is therefore bit-for-bit unchanged, counts
> as "untouched" or as "the first blended one". It is untouched. Exercise 2 settles it by
> comparing the computed `inv_freq` against the unscaled one element by element — do that
> before you trust either number, including this one.

In token terms: any dial with wavelength under about 256 tokens keeps its original speed;
any dial with wavelength over 8192 tokens is fully interpolated. That is the whole idea —
**preserve the geometry the model uses for local work, re-address only the range it has
never used.**

**What `β_slow = 1` actually means, and it is not obvious from the paper.** `β_slow` is
denominated in rotations, so `β_slow = 1` puts the fully-interpolated boundary exactly at
"the dial that completes one rotation in the training window". And full interpolation
divides `ω` by `s` while the window multiplies by `s`, so those dials complete *the same
number of rotations* in the extended window as they did in training. Consequence, which
you can check in Exercise 2: the set of dials completing at least one rotation inside
8192 with the original frequencies is `i = 0…17`, and the set completing at least one
rotation inside 1,048,576 with the YaRN-corrected frequencies is **also `i = 0…17`**
`[M]` (dial 17 blended down to `λ = 56 710`, so 18.5 rotations; dial 18 interpolated up to
`λ = 1 291 059`, so 0.81 rotations — exactly its training value). YaRN's band split is
constructed so the active dial set is invariant under extension. That is the actual
design claim, and it is arithmetic rather than intuition.

*(b) The attention temperature.* YaRN also multiplies the pre-softmax logits, on the
argument that attending over `s`× more keys raises the softmax entropy and needs
counteracting. The prescription is

```
attention_factor  =  0.1 · ln(s) + 1
```

For `s = 128`: `ln 128 = 4.852030263919617`, so `0.1 · ln 128 + 1 = 1.4852030263919618`
in float64 — **the shipped constant, to every digit** `[M]` (computed 2026-07-26;
`config.json:50` carries `1.4852030263919618`). For Laguna-XS with `factor: 32` the same
formula gives `1.3465735902799727`, also carried verbatim `[M]`.

Where it is applied matters and is easy to miss. `attention_scaling` multiplies **cos and
sin** (`modeling_laguna.py:122-123`), so it scales *both* the rotated query and the
rotated key, and therefore multiplies the logit by the **square**:

```
1.4852030263919618²  =  2.2058280296
```

**Every attention logit on the 12 full-attention layers is multiplied by 2.206.** That is
a temperature of `1/2.206 = 0.453` — a substantially sharper softmax — and it appears in
the config as a single unexplained float with no unit and no comment.

This is the cleanest example in the whole recipe of **inherited convention masquerading
as a tuned value**: it is the paper's suggested default, computed from `factor` alone,
with no search. It is a free ablation axis sitting in plain sight.

### 2.8 Partial rotary: two effects, not one

`partial_rotary_factor = 0.5` sets `rotary_dim = int(128 × 0.5) = 64`. In
`apply_rotary_pos_emb` (`modeling_laguna.py:295-305`) the head is split, the first 64
channels are rotated, the last 64 are concatenated back untouched. So the logit is

```
q · k  =  ⟨ R_m q_rot , R_n k_rot ⟩   +   ⟨ q_pass , k_pass ⟩
             function of (content, δ)        function of content ONLY
```

**Effect one — half of every global head is NoPE.** Not "weakly positional": those 64
channels carry no positional signal whatsoever. Laguna's global layers are a per-channel
RoPE/NoPE hybrid at exactly 50/50, and the model card's phrase "per-layer-type rotary
scales" does not say so; you have to work it out from `partial_rotary_factor`.

**Effect two — the frequency comb gets coarser, and nobody mentions this one.** The
frequencies are spread across `d_rot`, not across `head_dim`. Adjacent dials differ in
wavelength by the ratio `θ^{2/d_rot}`:

```
d_rot = 64,  θ = 500000   →   500000^{1/32}  =  1.507×  between adjacent dials
d_rot = 128, θ = 500000   →   500000^{1/64}  =  1.228×  between adjacent dials
```

`[M]` computed 2026-07-26. Halving the rotary fraction at fixed `θ` does not just delete
positional channels; it **doubles the log-spacing of the ones that remain**, so the
distance code is sampled half as densely across the same span. If you sweep
`partial_rotary_factor` as an ablation axis you are moving two things at once, and any
result that does not separate them is unattributable. That matters for us specifically:
our house rule is to instrument for attribution, and this axis is confounded by
construction unless you co-vary `θ` to hold the comb spacing fixed.

> **Systems bridge.** Partial rotary looks like splitting a record into a keyed field and
> an unkeyed field — half the head is a positional index, half is a content-only match.
>
> **Where it breaks:** they are not two lookups whose results you can inspect separately.
> The two inner products are **summed into a single scalar** before the softmax ever sees
> them. You cannot ask which half produced a match, and there is no way to instrument it
> without recomputing attention with one half zeroed. Attribution is destroyed by the sum,
> not by a missing counter.

### 2.9 NoPE, stated honestly

No rotation, no bias, no table. The logit is pure content, and ordering has to be
reconstructed from the nested structure of the causal mask (§1.2). The controlled
head-to-head — APE, T5-relative, ALiBi, RoPE, and nothing — found the useful *negative*
result: RoPE and ALiBi are not chosen because they extrapolate well, and NoPE does at
least as well as the schemes designed for the job `[C]`
([2305.19466](https://arxiv.org/abs/2305.19466), 2023). NoPE also has a ceiling,
attributed to "distraction of attention distributions" and fixable by tuning per-head
attention **temperature** `[C]` ([2404.12224](https://arxiv.org/abs/2404.12224), Apr
2024) — which is the same lever YaRN's `attention_factor` pulls globally and blind.

The 2026 fashion is the *layer-granular* version of Laguna's dimension-granular choice:
Cohere's RNoPE alternates RoPE and NoPE across attention blocks, with RoPE carrying local
context and NoPE carrying long-range retrieval `[C]`
([2501.18795](https://arxiv.org/abs/2501.18795), Jan 2025); SWAN-GPT interleaves NoPE
global layers with RoPE sliding-window layers `[C]`
([2504.08719](https://arxiv.org/abs/2504.08719), Apr 2025); DroPE *removes* positional
embeddings from an already-pretrained model to extend context `[C]`
([2512.12167](https://arxiv.org/abs/2512.12167), Dec 2025).

Note the direction is opposite to intuition in every case: the **global** layers get the
weakened positional signal and the **local** layers keep full, unscaled RoPE. That is
coherent once you see that a 512-token window never needs to represent a distance beyond
512 — and it is the reason "just widen the SWA window to test long context" is not a
valid experiment.

> **Systems bridge.** NoPE is soft state derived from the shape of the log rather than
> hard state stamped into the record — inferring commit order from a monotonically
> growing set instead of reading a sequence number.
>
> **Where it breaks:** a sequence number survives reordering, gaps and loss. The
> causal-mask signal is only a *count*, and a count degrades into an increasingly weak
> prior as it grows. That is exactly the "distraction" failure measured in 2404.12224.

### 2.10 The dtype floor on position arithmetic

One piece of arithmetic that costs nothing to check and settles an argument.

bfloat16 has 8 bits of significand. Integers are exactly representable up to 2⁸ = 256; in
the interval `[8192, 16384)` the spacing between representable bf16 values is
`2¹³ · 2⁻⁷ = 64`. **In bf16, positions 8192 and 8200 are the same number.** fp16 (11-bit
significand) is exact only to 2048. fp32 (24-bit) is exact to 16,777,216, which covers
Laguna's advertised 1,048,576 with room.

That is why `modeling_laguna.py:119-120` forces the `inv_freq @ position_ids` matmul into
fp32 under an explicit autocast disable. It is not defensive style; without it the
positional code does not exist above ~8k.

The residual exposure is the **cast back**: `cos` and `sin` are computed in fp32 and
returned in the model dtype (`modeling_laguna.py:125`). bf16 values in `[−1, 1]` are
quantized to roughly `2⁻⁸ ≈ 0.004`, so the composition identity of §2.3 — which is what
makes RoPE relative — is only approximately true, and the deviation depends on absolute
`m`, not just on `δ`. `[C]` ([2411.13476](https://arxiv.org/abs/2411.13476), Nov 2024)
reports that this breaks RoPE's relative-position property in long-context training with
error accumulating as sequence length grows and concentrated on the first token.

`[M]` Our own bf16 numerics on gfx1151 are **unproven** — `ASSUMPTIONS.md →
bf16-numerics-unproven` is `untested`, and RoPE at long positions is not currently on
that row's op list. Exercise 1 puts it there.

---

## 3. Why this matters for Proteus

### 3.1 The config surface, and one hard rule

Every ablation axis is a config field, so the positional scheme must be expressed as
fields, not as code branches. The shape is dictated by the reference model: **RoPE
parameters are per-layer-type, not per-model.**

```yaml
positional:
  scheme_by_layer_type:
    full_attention:
      kind: yarn                    # default | yarn | none
      theta: 500000.0
      partial_rotary_factor: 0.5
      yarn:
        factor: 128.0
        original_max_position_embeddings: 8192
        beta_fast: 32.0
        beta_slow: 1.0
        attention_factor: null      # null => 0.1*ln(factor)+1, and the RESOLVED value is logged
    sliding_attention:
      kind: default
      theta: 10000.0
      partial_rotary_factor: 1.0
```

Four design rules fall directly out of §2, each with a reason you can point at:

1. **There is no top-level `partial_rotary_factor`. Ever.** In transformers,
   `standardize_rope_params` writes the top-level scalar into *every* per-layer-type dict
   with a plain assignment, not a `setdefault`
   (`modeling_rope_utils.py:767-768`). Laguna's model code has to defend against it by
   hand (`modeling_laguna.py:595-606`) — a deep-copied config whose top-level field is
   re-aligned to the SWA value, with a five-line comment explaining that otherwise "the
   global partial factor silently clobbers the SWA one". A global overriding a per-shard
   setting, with no error and no log line: you have debugged this exact shape before, in
   Helm. We do not reproduce it.
2. **`kind: none` is a first-class value**, not "theta = infinity" or
   "partial_rotary_factor = 0". Layer-granular NoPE versus dimension-granular partial
   rotary is one of the live open questions (§7), and it cannot be an experiment if one
   arm is expressible only as a degenerate value of the other.
3. **`attention_factor: null` must resolve, and the resolved value must land in the run
   metrics.** An inherited default that is invisible in the config is invisible in the
   result. `2.206` is not a rounding detail; it is a 2.2× multiplier on every logit in the
   global layers.
4. **`theta` and `partial_rotary_factor` are co-varied or the sweep is unattributable**
   (§2.8). A `partial_rotary_factor` sweep at fixed `theta` moves the comb spacing too.
   The rig should either sweep them jointly or record the derived comb spacing
   `theta^(2/d_rot)` as a first-class metric so the confound is at least visible in the
   aggregation.

### 3.2 The Mnemosyne interface constraint

This is the part of the module that changes a package boundary, so it is worth stating as
a specification.

`modeling_laguna.py:642-645` derives positions for new tokens from the cache:

```python
past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
position_ids = (torch.arange(inputs_embeds.shape[1], device=...) + past_seen_tokens).unsqueeze(0)
```

Now suppose a Mnemosyne eviction policy has dropped 100 entries from the middle of the
cache. `get_seq_length()` returns the number of *stored* entries. The next token is
therefore rotated to a position 100 lower than the tokens still resident behind it — the
query is asking about a distance that is wrong by 100 for every surviving key. Nothing
raises. Nothing logs. Perplexity moves a little and you spend two days blaming the
policy's scoring function.

There are exactly **two self-consistent repairs**, and the bug is mixing them:

- **`preserve_original`** — keep each surviving entry's original absolute position, and
  derive the new token's position from the highest position ever issued, not from the
  cache length. Distances to survivors stay true; the position sequence has holes. This is
  what an H2O/SnapKV-style importance evictor implicitly assumes.
- **`renumber_compact`** — renumber every surviving entry by its index *within the cache*
  and give the new token position `len(cache)`. Distances become fictitious but dense and
  in-distribution. This is StreamingLLM's deliberate choice, and it is *why* StreamingLLM
  works past the training length `[C]`
  ([2309.17453](https://arxiv.org/abs/2309.17453), 2023): the renumbering keeps every
  position inside the trained range.

Two consequences for the package boundary:

- **Position must be an explicit argument to the cache write path**, never inferred from
  cache length. `MnemosyneCache.append(k, v, positions=...)`. Inferring it is the bug, and
  inferring it is also exactly what would couple Mnemosyne to a model's notion of "how
  long am I", which the boundary rule forbids anyway.
- **Every eviction policy declares a `position_policy` field** with those two values and
  no default. A policy that does not state one is not runnable. This is cheap to enforce
  and it converts a silent corruption into a config error.

### 3.3 The capacity arithmetic this module implies

`[M]` From `ASSUMPTIONS.md → kv-per-token-laguna` and
`research/memory/long-context-behavior.md` §4: `num_key_value_heads = 8` is uniform, so
KV cost is exactly `2 · 8 · 128 · 2 B = 4 KiB` per token per layer. The 36 sliding layers
are window-bounded at 512 and cost `4 KiB × 512 × 36 = 72 MiB` total, constant. The 12
global layers cost 48 KiB/token and grow.

At the advertised 1,048,576 positions that is **48.0 GiB + 72 MiB for one sequence**, bf16,
uncompressed. Against our measured `[M]` **≥62 GiB** fast tier at ~200 GB/s
(`notebook/uma-carveout-controls-fast-tier.md`), that fits — barely, before any weights are
resident. And it cannot be one tensor: `[M]` single buffers ≥32 GiB hang or fault on this
machine (`ASSUMPTIONS.md → large-tensor-fault-32gib`), so a full-length KV cache must be
chunked into at least two allocations by construction. **The whole 1M reach is carried by
12 of 48 layers**, and that hybrid is the difference between "impossible" (192 GiB
all-global) and "barely".

### 3.4 The ablation shortlist this module hands to Themis

From the survey's open questions, restated as arms:

| Arm | What it tests | Field(s) moved |
|---|---|---|
| `proteus-rope-partial-{1.0, 0.5, 0.25, 0.0}` | is 0.5 an optimum or an inheritance? | `partial_rotary_factor` (co-vary `theta`) |
| `proteus-nope-global` vs `proteus-partial-global` | dimension-split vs layer-split positional weakening — shipping in 2026, never compared | `kind` per layer type |
| `proteus-yarn-ramp-only` | YaRN with `attention_factor` forced to 1.0 — the attribution arm nobody publishes | `attention_factor: 1.0` |
| `proteus-yarn-temp-only` | temperature with the ramp disabled (`beta_fast = beta_slow`) | `beta_*` |
| `proteus-beta-sweep` | do `β_fast=32 / β_slow=1` survive re-derivation at a different `θ` and `L_orig`? | `beta_fast`, `beta_slow` |

---

## 4. Read the code

Paths are relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. Line numbers are pinned to the revisions in
`PROVENANCE.md`.

### 4.1 The baseline you are moving away from

| Where | What to look at, and why |
|---|---|
| `training/nanogpt/model.py:128`<br>`wpe = nn.Embedding(config.block_size, config.n_embd)` | A learned absolute position table with exactly `block_size` rows. This is the entire positional scheme of GPT-2, and it is one line. |
| `training/nanogpt/model.py:173` | The bounds check: `assert t <= self.config.block_size`. Note what is *not* here — no fault-in, no growth path, no fallback. Compare with your intuitions about a fixed-size ring buffer, then remember rows past the cap are trained-on-nothing rather than empty. |
| `training/nanogpt/model.py:178` | `pos_emb = self.transformer.wpe(pos)`, then added to the token embedding on the next line. **Position enters the residual stream, not the attention logit.** This is the additive scheme whose four-term expansion you did in §2.1 — read the two lines and then convince yourself the cross terms are unavoidable. |

### 4.2 RoPE itself

| Where | What to look at, and why |
|---|---|
| `models/laguna-s/modeling_laguna.py:69` | `class LagunaRotaryEmbedding` — the whole positional machinery is one small `nn.Module` that produces `(cos, sin)` and nothing else. It has no parameters. |
| `models/laguna-s/modeling_laguna.py:107` | The frequency ladder: `1.0 / base ** (arange(0, dim, 2) / dim)`. This is `ω_i = θ^(−2i/d_rot)` from §2.4 written in one expression. Note `dim` here is already `head_dim × partial_rotary_factor` (line 106). |
| `models/laguna-s/modeling_laguna.py:119` | `with maybe_autocast(..., enabled=False):  # Force float32`. This is §2.10 as a code comment. Ask yourself what breaks without it, then compute the bf16 spacing at position 8192. |
| `models/laguna-s/modeling_laguna.py:120` | `freqs = inv_freq_expanded @ position_ids_expanded` — the outer product that turns (32 frequencies) × (T positions) into a T×32 angle table. The whole positional computation is one matmul. |
| `models/laguna-s/modeling_laguna.py:122` | `cos = emb.cos() * self.attention_scaling`. **This is where YaRN's temperature is applied** — to `cos`, therefore to both `q` and `k`, therefore to the logit *squared*. If you were looking for the temperature in the attention function, it is not there. |
| `models/laguna-s/modeling_laguna.py:125` | The cast back to model dtype. Everything after this line is bf16, including the trigonometry that carries the relative-position property. |
| `models/laguna-s/modeling_laguna.py:263` | `rotate_half` — the split-half pairing convention of §2.5, three lines. |
| `models/laguna-s/modeling_laguna.py:300` | `q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)` — the rotation, executed. Verify by hand that this is `R(mω)` applied to the pair `(j, j + d_rot/2)`. |

### 4.3 Partial rotary

| Where | What to look at, and why |
|---|---|
| `models/laguna-s/modeling_laguna.py:295` | `rotary_dim = cos.shape[-1]` — the rotated width is *inferred from the shape of `cos`*, not read from config. A subtle coupling: change how `inv_freq` is built and you change how much of the head rotates, at a distance. |
| `models/laguna-s/modeling_laguna.py:296` | `q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]` — the 50/50 split on Laguna's global layers. `q_pass` is the NoPE half. |
| `models/laguna-s/modeling_laguna.py:304` | The concatenate-back. Look at what is *not* here: no separate scaling, no separate norm, no marker. Downstream code cannot tell the two halves apart. |

### 4.4 YaRN

| Where | What to look at, and why |
|---|---|
| `architecture/transformers/src/transformers/modeling_rope_utils.py:327` | `_compute_yarn_parameters` — the entire method, ~130 lines including docstring. Read the docstring first; it is the best available specification of what every YaRN config key does. |
| `architecture/transformers/src/transformers/modeling_rope_utils.py:402` | `if factor is None:` — `max_position_embeddings` is used **only** in this fallback. Laguna sets `factor: 128.0` explicitly, so the advertised 1,048,576 never enters the frequency computation at all. The "1M context" is bookkeeping, not an input. |
| `architecture/transformers/src/transformers/modeling_rope_utils.py:408` | `return 0.1 * mscale * math.log(scale) + 1.0` — the temperature formula. Compare against `config.json:50` and confirm to the last digit. |
| `architecture/transformers/src/transformers/modeling_rope_utils.py:425` | `find_correction_dim` — the inversion you derived in §2.7. Confirm it returns a *pair* index. |
| `architecture/transformers/src/transformers/modeling_rope_utils.py:436` | `linear_ramp_factor` — the clamped blend, including the `max += 0.001` singularity guard for `low == high`. |
| `architecture/transformers/src/transformers/modeling_rope_utils.py:451` | `low, high = find_correction_range(...)` — the call that produces `9` and `18` for Laguna. Note it is passed `original_max_position_embeddings` (8192), not `max_position_embeddings`. |
| `architecture/transformers/src/transformers/modeling_rope_utils.py:455` | The blend itself. Trace which of `inv_freq_extrapolation` / `inv_freq_interpolation` wins at `i = 0` and at `i = 31`, then count the untouched dials yourself — this is the 10-vs-9 discrepancy from §2.7. |

### 4.5 Per-layer-type dispatch, and the footgun

| Where | What to look at, and why |
|---|---|
| `models/laguna-s/config.json:42` | `rope_parameters` as a dict **keyed by layer type**. Two completely different positional schemes in one model. Read `:51` (`partial_rotary_factor: 0.5`) against `:56` (`1.0`). |
| `models/laguna-s/configuration_laguna.py:193` | Where the nested `rope_parameters["sliding_attention"]` is lifted out into a separate `swa_rope_parameters` attribute — with a comment saying that without it "the sliding-window layers silently reuse the full-attention rope." |
| `models/laguna-s/modeling_laguna.py:584` | The full-attention path: a `copy.deepcopy` of the config with `rope_parameters` flattened to the `full_attention` sub-dict, because `LagunaRotaryEmbedding` inherits Qwen2MoE's flat-shape contract. Per-layer-type RoPE is **not** a first-class abstraction anywhere in the stack; it is two config clones. |
| `models/laguna-s/modeling_laguna.py:595` | Read this five-line comment in full. It documents that `standardize_rope_params` unconditionally overwrites the per-layer-type `partial_rotary_factor` with the top-level scalar. |
| `architecture/transformers/src/transformers/modeling_rope_utils.py:767`<br>and `:768` | The offending lines, in the framework: the guard at `:767`, then at `:768` a plain assignment `rope_parameters[layer_type]["partial_rotary_factor"] = partial_rotary_factor` — inside the per-layer-type branch that begins at `:762`. Compare with the `setdefault` calls on `:765-766`. **This is a global silently overriding a per-shard setting.** It is the single most transferable lesson in this file for someone who has run Kubernetes. |
| `models/laguna-s/modeling_laguna.py:664` | `global_pe = self.rotary_emb(...)`, then `:670` builds a `{layer_type: (cos, sin)}` map and `:676` selects per layer. Two `(cos, sin)` tensors are computed per forward pass, not one. |
| `models/laguna-s/modeling_laguna.py:642` | `past_seen_tokens = past_key_values.get_seq_length()` — the line §3.2 is about. **Positions are derived from cache length.** Stare at this until the eviction hazard is obvious. |

### 4.6 The same design in C++, which is a better spec than the Python

| Where | What to look at, and why |
|---|---|
| `architecture/llama-cpp-laguna/src/models/laguna.cpp:184` | `n_rot_l` — the start of the per-layer RoPE divergence. Seven consecutive lines each choose a RoPE parameter by `is_swa_il`. This is the clearest statement anywhere of "two positional schemes, selected by a bit". |
| `architecture/llama-cpp-laguna/src/models/laguna.cpp:187` | `ext_factor_l = is_swa_il ? 0.0f : ext_factor` — YaRN turned off for sliding layers by zeroing the extension factor, which is how llama.cpp expresses "plain RoPE". |
| `architecture/llama-cpp-laguna/src/models/laguna.cpp:193` | `attn_factor_l = is_swa_il ? 1.0f : attn_factor`, with a comment explaining that `llama_context` pre-divides the attention factor to cancel ggml's internal mscale. Two frameworks, two different places the same constant is applied, one of them cancelling itself — worth reading before you trust any cross-framework logit comparison. |

---

## 5. Exercises

All three run on the Z13 (gfx1151, native Windows, one GPU). Activate with
`. .\scripts\activate-lab.ps1` from the repo root. Every exercise has a CPU path, and
Exercise 2 needs no GPU at all.

**ROCm/WSL2 caveat:** do not run these under WSL2. `[C]` ROCm issue #6022 clamps the ROCm
pool to the `.wslconfig` value; none of these exercises are large enough to hit it, but
the numerics wheel differs and the point of Exercise 1 is numerics. Native Windows,
`C:\venvs\lab`, the pinned wheel `torch 2.12.0a0+rocm7.13.0a20260313` `[M]`
(`ENVIRONMENT.md`).

---

### Exercise: measure where bf16 breaks RoPE's relative-position identity

**Difficulty 2/5. Writing: 45–75 min. Runtime: under a minute.**

RoPE's entire justification is the identity you proved in §2.3:

```
⟨ R_m q , R_n k ⟩  =  ⟨ q , R_{n−m} k ⟩
```

In exact arithmetic the two sides are equal for every `m`. In floating point they are not,
and `[C]` ([2411.13476](https://arxiv.org/abs/2411.13476), Nov 2024) predicts the gap
**grows with absolute `m` at fixed distance `δ`** — a length-dependent failure that a
fixed-shape numerics test cannot see.

**Build.** A ~40-line script (put it under `notebook/` while it is a one-off; it migrates
into Themis with tests if you reuse it):

1. Draw one random `q` and one random `k` of width `d_rot = 64` in float64, seeded.
2. Implement `apply_rope(x, position, theta, dtype)` yourself — do not import
   transformers. Build `inv_freq` per §2.4, compute the angle, take `cos`/`sin` in
   float64, then cast `cos`/`sin` to the target dtype exactly as
   `modeling_laguna.py:125` does, and do the `rotate_half` multiply in that dtype.
3. For `m` in a log-spaced sweep `{0, 10², 10³, 10⁴, 10⁵, 10⁶}` and fixed `δ = 64`:
   compute `A = ⟨rope(q, m), rope(k, m+δ)⟩` and `B = ⟨q, rope(k, δ)⟩`, both in the target
   dtype, and record `|A − B|` and `|A − B| / |B_fp64|`.
4. Repeat for `dtype ∈ {float64, float32, bfloat16}` and for `θ ∈ {10000, 500000}`.

**Deliverable — a number and a plot.** The number: the smallest `m` at which the bf16
relative error exceeds 1%. The plot: log-log, relative error against `m`, six curves.

**What to expect and how to read it.** float64 should sit at machine epsilon and stay
there — if it does not, your implementation is wrong, not the hardware. float32 should
degrade slowly: at `m = 2²⁰` the ULP of the fp32 angle is `2²⁰ · 2⁻²³ = 0.125` radians, so
the fastest dial's phase is only known to about 2% of a rotation. bf16 should degrade far
sooner and far worse.

**Then run the same script on CPU and on GPU and diff the numbers.** This is the part that
belongs to us rather than to the paper. `ASSUMPTIONS.md → bf16-numerics-unproven` is
`untested` on gfx1151, and RoPE at long positions is not on that row's op list. If CPU bf16
and GPU bf16 agree, you have measured a property of bf16 and should append RoPE to the
row's op list as `supported`. **If they disagree, you have found a hardware bug and should
stop and write it up** — that is a Hardware Validation Gate finding, not a curriculum
exercise result.

**CPU fallback:** identical, `device='cpu'`. Torch supports bf16 elementwise and matmul on
CPU; it is slower and irrelevant at these shapes. Running both is the point anyway.

**Trap:** `torch.arange(0, dim, 2)` must be integer before the division, exactly as
`modeling_laguna.py:108` does with `dtype=torch.int64`. Building the exponent in the target
dtype contaminates the frequency ladder itself and you will measure your own bug.

---

### Exercise: reconstruct Laguna's YaRN ladder from `config.json` and check all four published constants

**Difficulty 2/5. Writing: 60–90 min. Runtime: seconds. No GPU required.**

Everything YaRN does to Laguna is determined by six numbers in a JSON file. Recompute all
of it from scratch and check it against the framework.

**Build.** Read `research/reference/models/laguna-s/config.json` — parse it, do not
hardcode. Then, in plain Python with `math` and `torch`:

1. **The temperature.** Compute `0.1 * math.log(factor) + 1.0` in float64 and compare
   `repr()` against the config's `attention_factor`. Then compute its **square** and state,
   in one sentence, what that number multiplies.
2. **The band edges.** Implement `find_correction_dim` from §2.7 yourself and evaluate it
   at `β_fast = 32` and `β_slow = 1`. Apply `floor` and `ceil` as
   `modeling_rope_utils.py:431-433` does.
3. **The ladder.** Build `inv_freq` three ways: unscaled (`ω_i`), fully interpolated
   (`ω_i / 128`), and YaRN-blended per `modeling_rope_utils.py:455`. Compare element by
   element with `torch.equal` against the unscaled version and **count how many dials are
   bit-for-bit unchanged.** This settles the 10-vs-9 question in §2.7 empirically — record
   which it is.
4. **The plot.** Log-y wavelength against pair index, four series on one axis: Laguna
   sliding (`θ=10000, d_rot=128`), Laguna global unscaled, Laguna global YaRN-blended, and
   the pure-PI counterfactual (`ω/128` on every dial). Mark 512 (the SWA window), 8192
   (the training length) and 1,048,576 (the advertised context) as horizontal lines.
5. **The rotation accounting.** For each dial compute `r_train = 8192 · ω_i / 2π` and
   `r_extended = 1048576 · ω_i^{new} / 2π`. Print the ratio. YaRN's design intent is that
   the fully-interpolated dials show a ratio of exactly 1.0 — they see the same phase range
   they saw in training. Confirm or refute that, and note what the blended band does
   (`i = 17` should come out around 15×).

**Deliverable — a table and a plot.** The table: 32 rows, one per dial, with `λ_i`,
`r_train`, treatment (untouched / blended / interpolated), `λ_i^{new}`, `r_extended`. The
plot as described.

**Cross-check:** import `_compute_yarn_parameters` from the local transformers clone, call
it on the real config, and `torch.allclose` your `inv_freq` against it. If they differ, the
interesting question is which one is wrong — read `:446-458` line by line before assuming
it is you.

**CPU fallback:** this exercise is CPU-only by construction. It is deliberately the
cheapest one and it is the one that makes the config readable forever after.

---

### Exercise: a length-generalization sweep at nanoGPT scale

**Difficulty 4/5. Writing: one evening. Compute: `[A]` 2–5 h GPU total for four arms,
medium confidence — the basis is nanoGPT's published ~3 min/run on one A100
(`training/nanogpt/README.md:51`) scaled by our `[M]` 20.9 TFLOPS bf16 at 8192³
(`ASSUMPTIONS.md → hipblaslt-config`), which is a bad estimator for a launch-bound 10.6M
model. Time the first arm and re-plan.**

The question: **at 10M parameters, does the published ordering of positional schemes under
length extrapolation reproduce at all?** If it does not, this rig cannot study length
generalization, and that is a result worth having before you spend a month on it.

**Build.** Start from `research/reference/training/nanogpt`, `shakespeare_char`, the
published GPU config (6 layers / 6 heads / 384 channels / `block_size` 256, ~10.6M params).
Replace `wpe` with a switchable positional module — this is the same plug-point Proteus
will need, so write it as if it were Proteus code:

- `learned_absolute` — the stock `wpe`, as the control.
- `rope_full` — RoPE, `θ = 10000`, `partial_rotary_factor = 1.0`.
- `rope_partial` — RoPE, `θ = 10000`, `partial_rotary_factor = 0.5`.
- `nope` — nothing at all.

Train each at `block_size = 256`, matched token budget, **≥3 seeds** (they are cheap here;
single-seed numbers are anecdotes by house standard and must be labelled as such). Then
evaluate validation loss at context lengths 256, 384, 512, 768, 1024 — beyond the training
length in every case after the first.

Then, on the `rope_full` checkpoint only, apply four **zero-shot** extension schemes at
eval time and re-score at 1024 (`s = 4`):

| Arm | What changes |
|---|---|
| none | raw extrapolation |
| PI | `ω_i → ω_i / 4` on every dial |
| NTK-base | `θ → θ · 4^(d/(d−2))` |
| YaRN | the §2.7 ramp, `β_fast = 32`, `β_slow = 1`, plus `attention_factor = 0.1·ln 4 + 1` |
| YaRN, ramp only | identical but `attention_factor = 1.0` |

**Deliverable — two plots and one table.** Plot A: val loss against eval context length,
four positional schemes, with confidence intervals over seeds. Plot B: the same for the
five extension arms on one checkpoint. Table: val loss at 1024 for all nine configurations.

**What to expect, honestly.** `learned_absolute` cannot run past 256 at all — the assert
at `model.py:173` fires, and that is the correct answer, not a bug to work around. RoPE
should degrade sharply past 256. NoPE should degrade more gracefully. Raw extrapolation
should be the worst of the five extension arms.

**But do not assume the published ordering reproduces.** These are character-level 10M
models on ~1 MB of Shakespeare, five orders of magnitude below where any of these papers
measured. If PI beats YaRN here, or if NoPE wins outright, that is a finding about our
scale — and per `ASSUMPTIONS.md → ablation-scale-sufficient` (`untested`) it is exactly the
kind of evidence that row needs. Pre-register with the G2 card before you run, and write it
up in `notebook/` either way. A falsified hypothesis is a successful experiment.

The `YaRN, ramp only` arm is the attribution arm: it separates the frequency ramp from the
attention temperature. As of this writing nobody publishes it (§7).

**CPU fallback:** use the published CPU recipe — 4 layers / 4 heads / 128 channels /
`block_size` 64 / 2000 iters, target val loss 1.88 `[M]`
(`training/nanogpt/README.md:85`). Scale the eval lengths to 64 / 96 / 128 / 192 / 256 and
`s = 4` accordingly. The curve shapes are the deliverable, not the absolute losses, so the
CPU version answers the same question at lower resolution.

**Trap:** when you extend the eval context you must also extend the attention mask and the
`position_ids`, and it is easy to accidentally extend one and not the other. Add an
assertion that `position_ids.max() == context_len - 1` before every eval batch. This is a
miniature of the §3.2 hazard and you will get it wrong at least once.

---

## 6. Self-check

1. A colleague says "sinusoidal position encoding is relative, because `p_{m+k}` is a
   linear function of `p_m`." The statement about `p` is true. Why does the conclusion
   about the attention logit not follow?

2. Laguna's sliding layers use `θ = 10000` over all 128 head dimensions; its global layers
   use `θ = 500000` over 64. Without running anything, which bank has the longer maximum
   wavelength, and which has more dials that complete at least one full rotation inside
   *their own* attention span?

3. A Mnemosyne eviction policy drops 100 entries from the middle of a Laguna KV cache.
   Trace what happens to `position_ids` for the next generated token in the HF forward
   pass, then name the two self-consistent repairs and say which one StreamingLLM chose.

4. `attention_factor` is `1.4852030263919618`. Where in the code is it applied, what does
   it multiply, and by what factor does the pre-softmax attention logit change?

5. Someone proposes ablating `partial_rotary_factor ∈ {1.0, 0.5, 0.25}` at fixed `θ` and
   attributing any difference to "how much positional signal the head gets." Name the
   second thing that moved, and give the arithmetic that quantifies it for `θ = 500000`.

6. Why is "widen Laguna's sliding window from 512 to 4096 and measure long-context recall"
   not a valid experiment? Give the reason in terms of wavelengths, not in terms of masks.

7. Laguna advertises 1,048,576 positions. Name the two places that number appears in the
   stack, and the one place it conspicuously does *not*.

---

## 7. What is still unsolved here

The honest frontier, drawn from `research/memory/long-context-behavior.md` §"Contested"
and `research/notes/transformer-state-of-the-art.md` §11, plus what surfaced while writing
this module.

**1. Which half of YaRN does the work.** The per-band frequency ramp and the attention
temperature are two unrelated mechanisms shipped as one method. No published ablation
separates them. The missing arm is "YaRN with `attention_factor` forced to 1.0", and it
costs one config field at our scale. This is the single cheapest unpublished experiment in
the module.

**2. `β_fast = 32` and `β_slow = 1` have never been re-derived.** They are defaults from a
2023 paper, applied here at `s = 128`, a scale factor four times larger than anything in
the original evaluation, and at `θ = 500000` rather than 10000. Whether the band edges
should move with `θ` and `s` is an open, cheap, two-dimensional sweep.

**3. Dimension-split versus layer-split positional weakening has never been compared.**
Laguna weakens position per-channel (`partial_rotary_factor 0.5`); RNoPE `[C]`
([2501.18795](https://arxiv.org/abs/2501.18795), Jan 2025), SWAN-GPT `[C]`
([2504.08719](https://arxiv.org/abs/2504.08719), Apr 2025) and Arcee Trinity weaken it
per-layer. Both are shipping in 2026. No head-to-head exists at any scale.

**4. What the low-frequency dials are actually for — and this one undercuts YaRN's
premise.** `[C]` ([2410.06205](https://arxiv.org/abs/2410.06205), Oct 2024, rev. May 2025)
finds that in Gemma 7B the *high* frequencies build robust positional attention patterns
while the model preferentially uses the *lowest* frequencies to carry what looks like
**semantic** rather than positional information. If that generalizes, then YaRN's ramp —
which divides exactly the 14 lowest-frequency dials by 128 and leaves the fast ones alone
(§2.7) — is scaling a semantic channel by construction, not repairing a positional one.
The two stories predict opposite things about which dials it is safe to interpolate, and
nobody has run the discriminating experiment. Treat "wavelength = distance resolution" as
a working model, not a fact.

**5. Whether one frequency schedule per layer type is even the right granularity.** `[C]`
AdaRoPE (arXiv [2607.19363](https://arxiv.org/abs/2607.19363), 2026) gives every attention
*head* learnable rotation frequencies and its own attention scaling, and reports beating
partial-RoPE and NoPE baselines. Laguna ships two schedules for 48 layers and hundreds of
heads. Head-level heterogeneity is uncharted and is a natural Proteus arm — but the paper
is weeks old and unreplicated; treat it as a lead, not a result.

**6. Whether rescaling RoPE is the right frame at all.** Three incompatible answers, all
from the last eight months. Jet-Long `[C]`
([2607.07740](https://arxiv.org/abs/2607.07740), Jul 2026): keep scaling, make it dynamic
and length-adaptive. DroPE `[C]` ([2512.12167](https://arxiv.org/abs/2512.12167), Dec
2025): delete the positional embeddings from the pretrained model instead. SWAN-GPT `[C]`
([2504.08719](https://arxiv.org/abs/2504.08719), Apr 2025): design them out from the start.
Randomized YaRN `[C]` ([2606.23687](https://arxiv.org/abs/2606.23687), Jun 2026) moves the
whole fix into training. LongRoPE2 `[C]`
([2502.20082](https://arxiv.org/abs/2502.20082), Feb 2025) argues the real culprit is
insufficient *training* of the high-`i` dials rather than the scaling rule at all. The
field has not adjudicated; the resolution probably depends on token budget, which is an
axis a small rig can actually attack.

**7. Whether position bias is architectural or correctable.** `[C]`
([2602.16837](https://arxiv.org/abs/2602.16837), Feb 2026) derives the U-shaped influence
profile from causal masking plus residual connections — i.e. it falls out of the
architecture and more long-context training will not remove it. `[C]`
([2406.16008](https://arxiv.org/abs/2406.16008), Jun 2024) recovers up to 15 percentage
points by calibrating attention bias at inference, implying it is substantially
correctable. Both live. Do not let a curriculum assert either.

**8. bf16 × RoPE on gfx1151 is unmeasured, and it is ours to measure.** `[C]`
([2411.13476](https://arxiv.org/abs/2411.13476)) predicts a length-dependent
relative-position failure. `ASSUMPTIONS.md → bf16-numerics-unproven` is `untested` and
lists matmul / softmax / RMSNorm / attention — **RoPE at long positions is not on that
list and belongs there.** Exercise 1 is the cheapest item on the Hardware Validation Gate
and it should run before any long-context training on this machine.

**9. Whether the effective-versus-advertised gap even reproduces below 300M params.**
Every measurement of it — RULER `[C]`
([2404.06654](https://arxiv.org/abs/2404.06654)), LongBench Pro `[C]`
([2601.02872](https://arxiv.org/abs/2601.02872)), ATLAS `[C]`
([2605.28079](https://arxiv.org/abs/2605.28079)) — is at frontier scale. If the phenomenon
does not appear at our scale, this rig cannot study it, and that negative is a result about
our own methodology worth publishing internally.

---

## 8. Answers to the self-check

**1.** Because the logit is a dot product of two *projected sums*, and it expands into four
terms (§2.1): content×content, content×position(n), position(m)×content, and
position(m)×position(n). The linearity of `p` under offset only constrains the fourth term,
and only if `W_qᵀ W_k` cooperates. The two cross terms are irreducibly functions of absolute
`m` and absolute `n` and cannot be rewritten as functions of `n − m`. Rotation avoids this
because it multiplies the whole vector rather than being summed into it, so no cross terms
are ever created.

**2.** The **global** bank has the longer maximum wavelength: `λ_max = 2π · θ^{1 − 2/d_rot}`
gives **2,084,765** tokens at `θ=500000, d_rot=64` versus **54,410** at
`θ=10000, d_rot=128` `[M]`.

For the second half, compare each bank against *its own* span. Sliding layers, span 512:
`λ_i ≤ 512` holds for `i ≤ 30.6`, so **31 of 64** dials complete a rotation inside the
window and the other 33 act as a near-linear monotone ramp. Global layers, span 8192 at
training: `λ_i ≤ 8192` holds for `i ≤ 17.49`, so **18 of 32**. Global layers at the
extended 1,048,576 with YaRN-corrected frequencies: still `i = 0…17`, i.e. **18 of 32** —
because `β_slow = 1` places the fully-interpolated boundary exactly at the one-rotation
dial, and interpolation by `s` against a window multiplied by `s` leaves the rotation count
unchanged. So the fraction of "active" dials is 48% versus 56%, and it is invariant under
YaRN extension by construction. The instructive part is that the two banks are tuned to
spans three orders of magnitude apart, which is the entire point of a per-layer-type
schedule.

**3.** `modeling_laguna.py:642` sets `past_seen_tokens = past_key_values.get_seq_length()`,
which now returns 100 fewer than the true number of tokens seen. `:643-645` then gives the
new token a position 100 too low, so every distance it computes against a surviving key is
wrong by 100 — and nothing raises, logs, or checksums. The two self-consistent repairs are
**`preserve_original`** (keep each survivor's original absolute position and issue the new
token's position from the high-water mark, accepting holes) and **`renumber_compact`**
(renumber every survivor by its index within the cache, accepting fictitious but dense
distances). StreamingLLM chose `renumber_compact` deliberately, and that choice is *why* it
works past the training length — every position stays inside the trained range `[C]`
([2309.17453](https://arxiv.org/abs/2309.17453)).

**4.** It is applied at `modeling_laguna.py:122-123`, multiplying **`cos` and `sin`** — not
the attention scores. Because scaled `cos`/`sin` are used to rotate both the query and the
key, the logit is multiplied by the **square**: `1.4852030263919618² = 2.2058` `[M]`. So
every logit on the 12 full-attention layers is scaled by about **2.21×**, equivalent to a
softmax temperature of 0.453. The value itself is YaRN's untuned default,
`0.1·ln(128) + 1`, reproducible in float64 to the last digit.

**5.** The **frequency-comb spacing** moved. Frequencies are laid out across `d_rot`, not
across `head_dim`, so the wavelength ratio between adjacent dials is `θ^{2/d_rot}`. At
`θ = 500000`: `500000^{1/32} = 1.507×` at `partial_rotary_factor = 0.5`, against
`500000^{1/64} = 1.228×` at 1.0 `[M]`. Halving the rotary fraction therefore both removes
half the positional channels *and* doubles the log-spacing of the survivors. To attribute
the result to one of those, co-vary `θ` to hold `θ^{2/d_rot}` constant, or record the comb
spacing as a metric so the confound is visible in aggregation.

**6.** Because the sliding layers' rotary schedule was never trained to reach past the
window. Their `θ = 10000` bank has dials whose wavelengths inside 512 tokens are the only
phases those layers have ever observed for short distances; widening the window to 4096
presents distances 512–4096 whose phase combinations the layers saw at *training* time only
as masked-out, never-attended positions. It is the §2.6(a) argument applied at a smaller
scale: out-of-distribution phase, not out-of-range index. Confirmed in the artifact three
ways — the nested config, the separate `swa_rotary_emb` at `modeling_laguna.py:607`, and
`laguna.cpp:184-196` forcing plain RoPE on SWA layers.

**7.** It appears (a) as `max_position_embeddings` at `config.json:17`, where it sizes masks
and caches, and (b) as the arithmetic consequence `8192 × 128 = 1,048,576` — the
pretraining length times the YaRN factor. It conspicuously does **not** appear in the RoPE
frequency computation: `_compute_yarn_parameters` reads `max_position_embeddings` only in
the `factor is None` fallback (`modeling_rope_utils.py:402-403`), and Laguna sets `factor`
explicitly, so the frequency ladder is built entirely from `original_max_position_embeddings
= 8192`. The "1M context" is a statement about how frequencies were rescaled, not a
measurement of anything.

---

## 9. Sources

**Read from the repo's own artifacts** `[M]`, revisions pinned in
`research/reference/PROVENANCE.md`. Paths relative to `research/reference/`:

- `models/laguna-s/config.json` — `max_position_embeddings` (:17), `sliding_window` (:41),
  `rope_parameters` split by layer type (:42-58), `partial_rotary_factor: 0.5` (:51).
- `models/laguna-s/modeling_laguna.py` — rotary module (:69), frequency ladder (:107),
  forced fp32 (:119-120), temperature applied to cos/sin (:122), cast back (:125),
  `rotate_half` (:263), rotary/pass split (:295-305), full-attention config clone (:584-590),
  the `standardize_rope_params` workaround comment (:595-606), SWA rotary (:607),
  position derivation from cache length (:642), per-layer-type dispatch (:664-680).
- `models/laguna-s/configuration_laguna.py` — SWA rope derived from the nested dict (:193),
  `partial_rotary_factor` injection (:199).
- `architecture/transformers/src/transformers/modeling_rope_utils.py` —
  `_compute_yarn_parameters` (:327), factor fallback (:402), mscale formula (:408),
  `find_correction_dim` (:423-425), `linear_ramp_factor` (:436), frequency construction
  (:446), correction range (:451), the blend (:454-455), per-layer-type standardization
  (:762) and the unconditional overwrite (:767).
- `architecture/llama-cpp-laguna/src/models/laguna.cpp` — per-layer RoPE divergence (:184),
  YaRN disabled on SWA (:187), attention factor forced to 1.0 on SWA (:193).
- `training/nanogpt/model.py` — learned position table (:128), the bounds assert (:173),
  additive position (:178). `training/nanogpt/README.md` — the GPU target (:51), the CPU
  recipe and its target (:85).
- `ASSUMPTIONS.md` — `gpu-fast-tier-size` (≥62 GiB at ~200 GB/s),
  `large-tensor-fault-32gib`, `bf16-numerics-unproven`, `kv-per-token-laguna`,
  `hipblaslt-config`, `ablation-scale-sufficient`.
- `notebook/uma-carveout-controls-fast-tier.md`, `ENVIRONMENT.md`.
- Survey notes this module must stay consistent with:
  `research/memory/long-context-behavior.md`, `research/memory/kv-cache-mechanics.md`,
  `research/notes/transformer-state-of-the-art.md` §5.

**Cited** `[C]`. Every arXiv id below was resolved against arxiv.org on 2026-07-26.

- [1706.03762](https://arxiv.org/abs/1706.03762) — Attention Is All You Need (2017).
  Sinusoidal absolute position.
- [1803.02155](https://arxiv.org/abs/1803.02155) — Self-Attention with Relative Position
  Representations (Mar 2018). The relative-bias family.
- [2104.09864](https://arxiv.org/abs/2104.09864) — RoFormer: Enhanced Transformer with
  Rotary Position Embedding (2021). RoPE.
- [2108.12409](https://arxiv.org/abs/2108.12409) — Train Short, Test Long: Attention with
  Linear Biases Enables Input Length Extrapolation (Aug 2021). ALiBi.
- [2306.15595](https://arxiv.org/abs/2306.15595) — Extending Context Window of Large
  Language Models via Positional Interpolation (Jun 2023). PI.
- [2309.00071](https://arxiv.org/abs/2309.00071) — YaRN: Efficient Context Window Extension
  of Large Language Models (2023).
- [2309.17453](https://arxiv.org/abs/2309.17453) — Efficient Streaming Language Models with
  Attention Sinks (2023). StreamingLLM; the in-cache renumbering choice.
- [2305.19466](https://arxiv.org/abs/2305.19466) — The Impact of Positional Encoding on
  Length Generalization in Transformers (2023). NoPE.
- [2404.12224](https://arxiv.org/abs/2404.12224) — Length Generalization of Causal
  Transformers without Position Encoding (Apr 2024).
- [2410.06205](https://arxiv.org/abs/2410.06205) — Round and Round We Go! What makes Rotary
  Positional Encodings useful? (Oct 2024, rev. May 2025).
- [2411.13476](https://arxiv.org/abs/2411.13476) — When Precision Meets Position: BFloat16
  Breaks Down RoPE in Long-Context Training (Nov 2024).
- [2501.18795](https://arxiv.org/abs/2501.18795) — Rope to Nope and Back Again: A New Hybrid
  Attention Strategy (Jan 2025, rev. Oct 2025). RNoPE.
- [2502.20082](https://arxiv.org/abs/2502.20082) — LongRoPE2: Near-Lossless LLM Context
  Window Scaling (Feb 2025).
- [2504.08719](https://arxiv.org/abs/2504.08719) — SWAN-GPT: An Efficient and Scalable
  Approach for Long-Context Language Modeling (Apr 2025).
- [2512.12167](https://arxiv.org/abs/2512.12167) — Extending the Context of Pretrained LLMs
  by Dropping Their Positional Embeddings (Dec 2025). DroPE.
- [2606.23687](https://arxiv.org/abs/2606.23687) — Randomized YaRN Improves Length
  Generalization for Long-Context Reasoning (Jun 2026).
- [2607.07740](https://arxiv.org/abs/2607.07740) — Jet-Long: Efficient Long-Context
  Extension with Dynamic Bifocal RoPE (Jul 2026).
- [2607.19363](https://arxiv.org/abs/2607.19363) — AdaRoPE: Not All Attention Heads Should
  Rotate and Scale Equally (2026). **Weeks old, unreplicated — a lead, not a result.**
- [2404.06654](https://arxiv.org/abs/2404.06654) — RULER (2024);
  [2601.02872](https://arxiv.org/abs/2601.02872) — LongBench Pro (Jan 2026);
  [2605.28079](https://arxiv.org/abs/2605.28079) — ATLAS (May 2026). Effective versus
  advertised context.
- [2406.16008](https://arxiv.org/abs/2406.16008) — Found in the Middle: Calibrating
  Positional Attention Bias (Jun 2024);
  [2602.16837](https://arxiv.org/abs/2602.16837) — A Structural Theory of Position Bias in
  Transformers (Feb 2026). The contested pair on whether position bias is correctable.
- ROCm issue #6022 (librocdxg VRAM mapping under WSL2) and #6034 (gfx1151 bf16 defects) —
  GitHub issues, not papers; cited as such in `CLAUDE.md` and `ASSUMPTIONS.md`.
