---
title: Quantization — two decisions, two error budgets, and a machine that pays for one of them
version: 1.0.0
date: 2026-07-26
track: F — Inference
mirrors: research/notes/inference-and-quantization.md §2–3; research/memory/kv-cache-mechanics.md §5
prereqs: tensors-and-autograd, attention-variants-and-kv-cost, kv-cache-mechanics, moe-and-routing
difficulty: moderate — one equation applied four ways. The hard part is refusing to let the two error budgets merge, and noticing that every "4-bit" format in production is not 4 bits.
time: 3–4 h reading and working the arithmetic; 1–2 h for the three exercises. **Exercises B and C were run on the Z13 and carry measured reference tables; Exercise A was not, because the model weights turned out to be Git LFS stubs — see the boxed note at the head of §6. Two of this module's own predictions were falsified by its own exercises; the corrections are in §3.8.**
---

# Quantization

**Prerequisites, and what this module refuses to re-teach.** You need
`tensors-and-autograd.md` for dtypes and what a `torch.dtype` actually commits you to;
`attention-variants-and-kv-cost.md` for the product `2·L·n_kv·d_h·b` and the
arithmetic-intensity result `AI = 2G/b`; `kv-cache-mechanics.md` for the three budgets
(residency / read traffic / maintenance traffic) and for the already-measured fact that
an fp8 KV cache on this machine runs **2.9–3.1× slower** than bf16 while holding half the
bytes `[M]`; and `moe-and-routing.md` for why "the model is 4-bit" is almost always a
statement about expert weights only. None of that is repeated here. This module is about
the `b` factor itself: where it comes from, what it costs, and why there are two of them.

---

## 1. What this module settles

**One.** Weight quantization and KV quantization are two separate decisions with two
separate error budgets, two separate consumers, two separate calibration stories, and no
shared knob — the field routinely reports them as one number ("W4A16KV4") and the
composition has essentially never been attributed, so this module keeps them apart from
the first equation onward.

**Two.** Every format is exactly three choices — bits per element, elements per shared
scale, and the scale's own dtype — and the third choice is where the advertised bit count
goes to die: read from the shipped source, `q4_K` is **4.5** bits/weight, `q8_0` is
**8.5**, MXFP4 is **4.25**, NVFP4 is **4.5**, and an int8 KV cache with per-token-head
scales costs **3.1%** over its nominal size while a 2-bit one costs **12.5%** `[M]`
(read from `ggml-common.h` and `kv_cache_interface.py` at the pinned revisions).

**Three.** On gfx1151 we get the *capacity* half of quantization and not the *arithmetic*
half — `[M]` `torch._scaled_mm` raises `RuntimeError: torch._scaled_mm is only supported on
CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+`, reproduced in a fresh
process for this module (§6 Exercise B), and no Marlin/CUTLASS low-bit kernel path exists
on ROCm `[M]` (read from `auto_awq.py:312`, `fp8.py:287`) — which makes this a worse
throughput lab and a cleaner memory-economics lab, and which means that until the Hardware
Validation Gate closes, **every quantization-error number we measure here is confounded
with unproven bf16 numerics** `[M]` (`ASSUMPTIONS.md → bf16-numerics-unproven`) and with a
hipBLASLt setting that we have already shown moves long-reduction error by **2.8×** `[M]`
(`ASSUMPTIONS.md → hipblaslt-config`).

---

## 2. Theory in plain language

### 2.1 The one operation, and what it replaced

Quantization is a lossy affine re-encoding of a tensor. You pick a scale, divide, round to
an integer (or to a low-precision float), store the integer, and multiply the scale back
in when you need the number. That is the whole mechanism. Everything else in the
literature — GPTQ, AWQ, SmoothQuant, QuaRot, KIVI, KVQuant, MXFP4, NVFP4 — is an argument
about **which numbers share a scale** and **what you do to the tensor before you pick it**.

What it replaced, in order:

- **fp32 everywhere** (through ~2017). Training and inference both.
- **fp16/bf16 mixed precision** (2017–2020). Halved everything; became the default and is
  still the *reference* precision — bf16 is what our `[M]` measured numbers are taken in.
- **int8 post-training quantization** (2020–2022), which promptly broke on transformers
  because of activation outliers, producing `[C]` LLM.int8() (2208.07339, Aug 2022) —
  split the outlier channels out into a separate fp16 matmul — and `[C]` SmoothQuant
  (2211.10438, Nov 2022) — migrate the difficulty from activations into weights with a
  per-channel rescale.
- **Weight-only 4-bit** (2022–2023): `[C]` GPTQ (2210.17323) and `[C]` AWQ (2306.00978).
  This is the regime that actually shipped, and W4A16 (4-bit weights, 16-bit activations)
  is the boring default in 2026.
- **Rotation** (2024–): `[C]` QuaRot (2404.00456) and `[C]` SpinQuant (2405.16406) —
  multiply by an orthogonal matrix first so that no single element dominates its block.
  This lineage is not confined to papers; it is in the reference engine, gated by a
  runtime heuristic nobody documents (`llama-kv-cache.cpp:319`).
- **Block floating point / microscaling** (2023–): `[C]` MX formats (2310.10537), giving
  MXFP4 and NVFP4, which are the current frontier and are **contested** (§8).

Two things to notice about that history. First, it is not a history of better rounding.
Every step is a different answer to "what shares a scale, and can I change the tensor so
that sharing hurts less." Second, the outlier problem appeared the moment transformers
got large and has never gone away — it is the single fact that organises the field.

### 2.2 Two decisions, not one

This is the module's spine. Weights and the KV cache are both "tensors you could store in
fewer bits," and they are otherwise almost nothing alike.

| | **Weight quantization** | **KV-cache quantization** |
|---|---|---|
| What is quantized | model parameters | activations, cached per request |
| When | once, offline, at build time | online, on the write path of the decode loop |
| Calibration data | yes — a few hundred sequences, offline, unlimited compute | **none** — you see one token at a time and cannot look ahead |
| Amortized by batch? | **yes** — one copy serves every request in the batch | **no** — each sequence owns its own cache |
| Grows with context? | no | linearly (global layers) |
| Error is | fixed at build time, identical for every request forever | different per request, and **accumulates over the generation** |
| Reversible? | yes — rebuild the artifact from the bf16 checkpoint | no — the bf16 original was never stored |
| The consumer | the GEMM in the MLP and the projections | the two matmuls inside attention |
| Failure surface | perplexity, then reasoning, then instruction-following | the same, plus long-context recall and a documented alignment effect |
| What it buys you | model *fits*; weight read traffic falls | context *length* and batch *width* |

The single most consequential row is **"amortized by batch."** Weight bytes are read once
per forward pass and divided across the whole batch; KV bytes are read once per sequence
per decode step and divided across nothing. That is derived in
`kv-serving-hierarchy.md` §1 and it means the two quantization decisions bind at
completely different operating points. At batch 1 with a short context, weights dominate
and weight quantization is the whole game. At batch 32 with a long context, the KV cache
dominates and weight quantization is nearly irrelevant to throughput while KV
quantization is nearly all of it.

> **Systems bridge.** You have shipped both of these. Weight quantization is
> **compression at rest**: you compress the artifact once, at build time, with the whole
> corpus in front of you, and every reader pays the same decode cost. KV quantization is
> **compression in flight**: a stream codec that must choose its parameters online, per
> stream, with no lookahead, on the hot write path, while the consumer is reading the
> same buffer.
>
> **Where it breaks — four places, and they are the module.**
>
> **Break 1 — lossy with no verify-on-read.** Every compressor you have operated is
> lossless, or lossy with a human in the loop who can see the artifact. Here the
> decompressed value is silently wrong and there is no checksum, no CRC, no "corrupt
> block" alert. The failure presents as a fluent, confident, wrong answer. This is the
> same structural point `kv-eviction-policies.md` makes about eviction, arriving from a
> different direction: **the system has no error signal for its own approximation.**
>
> **Break 2 — the error is structured, not white.** Compression ratio on a text corpus is
> a property of entropy and behaves smoothly. Quantization error here is dominated by a
> handful of channels whose magnitudes are one to two orders of magnitude above the
> median. Averages will not tell you anything. Every algorithm named in §2.1 is an
> outlier-management technique wearing a rounding technique's clothes.
>
> **Break 3 — the ratio you quote is not the ratio you get.** A "4-bit" format ships a
> scale alongside the payload, and the scale is not free. §3.3 turns this into a ledger.
> The analogy to keep is filesystem metadata overhead at small block sizes — except here
> shrinking the block is the *only* accuracy lever, so the overhead grows exactly as you
> push the format harder.
>
> **Break 4 — for KV, the axis you want is the axis you cannot stream along.** §3.6. This
> is the interesting one, and it has an exact analogue in columnar storage.

### 2.3 A footnote that will not stay a footnote: it is not two decisions, it is four

"Two decisions" is the right teaching frame and the wrong count. Read
`kv_cache.py:65–69` in vLLM: an attention layer carries **four** scale parameters —
`q_scale`, `k_scale`, `v_scale`, and `prob_scale`, where `prob_scale` quantizes
`P = softmax(QK^T)` before the `PV` matmul. So a fully quantized attention layer makes
four independent precision choices, one of which (the attention probabilities) is an
*intermediate that never leaves the kernel* and appears in no config file, no model card,
and no compression-ratio claim.

Hold that. It is the mechanism behind the most instructive published failure in this area:
`[C]` the vLLM engineering writeup (22 Apr 2026) reports a 128k needle-in-a-haystack task
falling from **91% to 13%** under FP8 KV *plus* FP8 attention on Hopper, traced not to the
storage format but to imprecise fp32 accumulation inside the tensor-core path, and
restored to **89%** by a two-level accumulation fix that kept the speed. The number format
was innocent; the kernel was guilty. If you report "FP8 KV cost us 78 points" you have
reported the wrong variable.

---

## 3. The math that actually matters

### 3.1 Symbols

Every symbol used below, translated. Carried symbols keep their meaning from Track B and
`kv-cache-mechanics.md`.

| Symbol | Reads as |
|---|---|
| `w` | one real-valued number to be stored (a weight, or one element of a K or V vector) |
| `ŵ` | the value you get back after storing and reloading it — the *dequantized* value |
| `b` | **bits** in the stored payload for one element (4, 8, 16). Careful: Track B's lowercase `b` is *bytes* per element; here `b` is bits and `B_e = b/8` is bytes. Stated explicitly because the two conventions collide and the collision has cost people a factor of eight |
| `g` | **group size** — how many consecutive elements share one scale |
| `s` | the **scale** for a group: the real number you multiply the stored integer by |
| `z` | the **zero-point** for a group: an integer offset, used only by asymmetric quantizers |
| `q` | the stored integer, `q ∈ [−2^(b−1), 2^(b−1)−1]` symmetric, or `[0, 2^b−1]` asymmetric |
| `Δ` | the **step size** — the gap between two representable values. For a symmetric integer quantizer `Δ = s` |
| `m_G` | `max_{i∈G} |w_i|` — the largest magnitude inside group `G`. The single quantity that determines that group's error |
| `σ` | standard deviation of the weights in the tensor |
| `L, n_kv, d_h, T, G_l` | layers, KV heads, head dim, context length, GQA group size — as in Track B |
| `c` | bytes per token per layer of KV = `2 · n_kv · d_h · B_e` |
| `x` | the activation vector a quantized weight matrix multiplies |
| `H` | the second-moment matrix `E[x xᵀ]` of the activations into a layer — the "Hessian" in the GPTQ literature |

### 3.2 The quantizer, written out

**Symmetric, per group `G`:**

```
s   = m_G / (2^(b−1) − 1)          scale: the biggest magnitude in the group,
                                    divided by the biggest integer you can store
q_i = clamp(round(w_i / s), −2^(b−1), 2^(b−1)−1)     store this
ŵ_i = s · q_i                                        get this back
```

In words: measure the loudest thing in the group, set the gain so it just fits, and store
everything relative to that gain.

**Asymmetric** adds a zero-point so the group's *range* rather than its *magnitude* is
what fits:

```
s = (max_G w − min_G w) / (2^b − 1)
z = round(−min_G w / s)
q_i = clamp(round(w_i / s) + z, 0, 2^b − 1)
ŵ_i = s · (q_i − z)
```

Asymmetric costs you a second per-group number (the zero-point) and buys you one bit of
effective range when the distribution is not centred. Weight distributions in trained
transformers are close to zero-mean, so symmetric is the usual choice for weights; AWQ's
shipped format is asymmetric (`auto_awq.py` carries `zero_point` as a config field), and
vLLM's int4 KV mode is described in its own enum comment as "packed 2×int4/byte, **RHT +
asymmetric zp**" (`kv_cache_interface.py:44`) — RHT being a randomized Hadamard transform,
i.e. the rotation trick from §2.1, in production, in a KV cache.

**The error, and the two things it depends on.** Round-to-nearest gives
`|w_i − ŵ_i| ≤ Δ/2 = s/2`, so the worst-case error for any element in the group is

```
                 m_G
|w − ŵ|  ≤  ─────────────────
             2 · (2^(b−1) − 1)
```

Read that as a sentence: **the error on every element in a group is set by the largest
element in that group.** One outlier at 100σ in a group of 128 raises the error floor for
the other 127 by a factor of 100 relative to what they would have suffered alone. That is
the entire outlier problem in one inequality, and it is why every advance in this field is
either "make the group smaller" or "make the largest element smaller."

If you model the rounding residual as uniform on `[−Δ/2, Δ/2]` — a standard and roughly
defensible assumption for a well-populated group — its variance is `Δ²/12`, giving a
signal-to-noise ratio

```
SNR = σ² / (Δ²/12) = 12 σ² (2^(b−1) − 1)² / m_G²
```

Take `10·log₁₀` of that and you recover the classic **≈6 dB per bit**: each extra bit
halves `Δ` and quarters the noise power. The useful form for our purposes is the ratio
`m_G/σ` — the **outlier ratio** — because it enters the SNR exactly as the bit count does.
An outlier ratio of 16 costs you the same 4 bits that dropping from int8 to int4 costs
you. Exercise A measures this ratio on the reference model's real weights.

### 3.3 The half-bit tax: effective bits, from the shipped source

Nominal bit counts are marketing. The real number is

```
                  payload_bits + scale_bits + zeropoint_bits
effective_bits =  ──────────────────────────────────────────
                              elements_per_block
```

`[M]` **Read from `research/reference/architecture/llama-cpp-laguna/ggml/src/ggml-common.h`
at the revision pinned in `PROVENANCE.md`.** Every row below is a C struct plus the
`static_assert` on the next line that pins its size; you can check the arithmetic by
opening the file, which is the point of putting it here rather than citing a table.

| Format | Line | Block layout | Bytes / block | Elements | **Effective bits/element** |
|---|---|---|---|---|---|
| `block_q8_0` | `:252` | 1 fp16 scale + 32 int8 | 34 | 32 | **8.5** |
| `block_q4_0` | `:195` | 1 fp16 scale + 16 B nibbles | 18 | 32 | **4.5** |
| `block_q4_1` | `:202` | fp16 scale + fp16 min + 16 B | 20 | 32 | **5.0** |
| `block_mxfp4` | `:215` | 1 uint8 E8M0 exponent + 16 B E2M1 | 17 | 32 | **4.25** |
| `block_nvfp4` | `:223` | 4 × UE4M3 sub-block scales + 32 B E2M1 | 36 | 64 | **4.5** |
| `block_q4_K` | `:327` | 2 fp16 super-scales + 12 B of 6-bit sub-scales + 128 B | 144 | 256 | **4.5** |
| `block_q2_K` | `:298` | 2 fp16 + 16 B 4-bit scales/mins + 64 B | 84 | 256 | **2.625** |
| `block_q2_0` | `:188` | 1 fp16 scale + 16 B | 18 | 64 | **2.25** |
| `block_q1_0` | `:181` | 1 fp16 scale + 16 B | 18 | 128 | **1.125** |
| `block_tq1_0` | `:276` | ternary, 5 values per byte | — | 256 | **1.6875** (source comment) |

Four readings worth taking from that table.

1. **Nothing is its nominal size.** The cheapest 4-bit format on the list is 4.25 bits and
   the most common one is 4.5. A 12.5% error in a capacity plan is the difference between
   fitting and not (§4.1).
2. **MXFP4 beats NVFP4 on bits and loses on accuracy, and the reason is visible in the
   struct.** MXFP4 shares one power-of-two exponent across 32 elements; NVFP4 shares an
   FP8 (UE4M3, i.e. non-power-of-two) scale across 16. Finer groups and a scale that can
   land between powers of two — that is the entire difference, and it costs exactly a
   quarter of a bit per element. `[C]` (2603.08747, Mar 2026, finds the sensitivity is
   strongly layer- and block-dependent rather than uniform.)
3. **The k-quants pay their scale overhead twice, and it still wins.** `q4_K` stores
   *quantized scales* — 12 bytes of 6-bit sub-scales for eight 32-element sub-blocks —
   under two fp16 super-scales. That is a two-level scale hierarchy, and its whole purpose
   is to buy `g = 32` granularity at the price of `g = 256` metadata. Same bits/element as
   `q4_0`, much better error, because §3.2 says granularity is the lever.
4. **The `static_assert` on the line after each struct is the contract.** This is a
   well-engineered wire format: the size is asserted at compile time, so a layout change
   that would silently corrupt every existing GGUF file fails the build instead.

**And a format fact that is not about bits at all: "FP8" names two different number
systems, and torch will hand you either.** `[M]` **Measured on our instrument in a fresh
process**, gfx1151, torch `2.12.0a0+rocm7.13.0a20260313`, deterministic (no seeds
involved — it is an enumeration of all 256 bit patterns; Exercise B is the script):

| Fact | Value |
|---|---|
| `element_size()` for `float8_e4m3fn`, `e5m2`, `e4m3fnuz`, `e5m2fnuz` | **1 byte** each |
| Distinct values of `value_as_e4m3fn / value_as_e4m3fnuz` over all bit patterns finite and non-zero in both (252 of 256) | **exactly `{2.0}` — one element, no spread** |
| Largest finite `e4m3fn` | **448.0** |
| Largest finite `e4m3fnuz` | **240.0** |
| Bit pattern `0x80` read as `e4m3fn` | **−0.0** |
| Bit pattern `0x80` read as `e4m3fnuz` | **NaN** |

Read those six rows together, because the pair of them is a trap.

The ratio row confirms `w8a8_utils.py:128` from first principles: the FNUZ variant used by
AMD's MI300-class parts has an exponent bias one larger than the OCP variant, so **for the
same eight bits it means half the number**, and any cross-format load must double every
scale. The `0x80` row confirms `w8a8_utils.py:116-117`: the byte that is a perfectly
ordinary negative zero in one system is a NaN in the other, so a byte-for-byte copy of a
checkpoint between the two does not merely scale wrong, it can poison the tensor.

Now the subtlety the code comment does not mention and the measurement does: **the maxima
are not in a 2:1 ratio.** 448 / 240 = 1.867, not 2. FNUZ spends no bit patterns on ±inf
and reserves only `0x80` for NaN, so it recovers the top mantissa step that OCP burns on
NaN encodings. So the two formats have the same *resolution* per bit pattern (ratio
exactly 2) and different *dynamic ranges* (ratio 1.867). If you calibrate a scale against
`448.0` and then store into `e4m3fnuz`, your top 7% of range clips silently.

`[M]` **And the reason this is a hazard here rather than trivia:** on gfx1151 all four
variants allocate happily at 1 byte/element, and nothing in the API stops you picking the
wrong one. vLLM's rule is a single string test — `return "gfx94" in _GCN_ARCH`
(`rocm.py:890`) — and we are not gfx94x, so **OCP `e4m3fn` is the correct choice on this
machine** and the FNUZ variants are a two-times-wrong footgun that runs without error.

> **Systems bridge.** Block size versus metadata overhead is the oldest tradeoff in
> storage: 512-byte sectors versus 4 KiB versus 1 MiB extents, inode overhead versus
> internal fragmentation, and the same log-shaped curve.
>
> **Where it breaks.** In a filesystem the block size trades *space* against *space* —
> small blocks waste metadata, large blocks waste slack — and correctness is unaffected
> either way. Here the block size trades space against **accuracy**, and the accuracy loss
> is unobservable at write time and unrecoverable afterward. There is no `fsck` for a
> group whose scale was set by one outlier.

### 3.4 Why the group is the only real lever, and what the alternatives to shrinking it are

From §3.2, group error ∝ `m_G`. Three families of response, and they are exhaustive:

**(a) Shrink the group.** `m_G` falls because fewer elements can contain the outlier.
Cost: `1/g` more scale bytes. This is why `g = 128` and `g = 32` are the two numbers you
see everywhere: `g = 128` costs `16/128 = 0.125` bits/element with an fp16 scale, `g = 32`
costs `0.5`.

**(b) Move the outliers somewhere else.** `[C]` LLM.int8() (2208.07339) keeps the outlier
*channels* in fp16 and runs a second, narrow matmul for them. `[C]` SmoothQuant
(2211.10438) exploits an exact algebraic identity — for a diagonal positive `S`,

```
x W  =  (x S⁻¹)(S W)
```

— to move difficulty from the activations (hard to quantize, dynamic, uncalibratable) into
the weights (easy, static, calibratable). Note what makes this legal: `S` is folded into
the *previous* layer's output scaling at build time, so it costs nothing at run time. This
is a change of coordinates, not an approximation.

**(c) Rotate so that no element is an outlier.** For an orthogonal `Q` (`QᵀQ = I`),

```
x W  =  (x Q)(Qᵀ W)
```

exactly. A random Hadamard `Q` mixes every coordinate into every other, so a single large
coordinate is spread across the whole block and `m_G` collapses toward the RMS. `[C]`
QuaRot (2404.00456) samples `Q`; `[C]` SpinQuant (2405.16406) learns it. The reason to
care operationally: a Hadamard matrix of size `2^k` has a `O(n log n)` transform and
entries `±1/√n`, so the rotation is cheap enough to run inline. That is exactly what the
reference engine does to K and V before a quantized cache store
(`llama-kv-cache.cpp:319`), gated by `ggml_is_quantized(type_k) && head_dim % 64 == 0`
with the Hadamard matrices precomputed at cache construction (`:344`) — a runtime
heuristic, in no model config, disableable by an undocumented `LLAMA_ATTN_ROT_DISABLE`
env var.

### 3.5 Weight error is not the objective. Output error is.

Minimizing `‖W − Ŵ‖` is the wrong problem, and this is the single conceptual step that
separates GPTQ/AWQ from naive rounding. What the model cares about is

```
minimize   E_x ‖ x W  −  x Ŵ ‖²   =   tr( (W − Ŵ)ᵀ H (W − Ŵ) ),      H = E[x xᵀ]
```

Every symbol: `x` is an activation row entering this layer, drawn from real data; `W` is
the true weight matrix; `Ŵ` the quantized one; `H` is the activation second-moment matrix
— for input channel `j`, `H_jj` is the mean squared magnitude of activations arriving on
that channel. The objective is a *weighted* error, and the weights are the activation
energies.

Two consequences, and they are the two shipped algorithms:

- **GPTQ** `[C]` (2210.17323) minimizes exactly this, greedily, one column at a time,
  pushing each column's rounding residual into the columns not yet quantized using the
  inverse Hessian. Its `desc_act` option — visible as a first-class config field at
  `auto_gptq.py:110` — quantizes columns in order of *decreasing* activation magnitude, so
  the important columns are decided while there is still error budget to redistribute.
  The engineering tell that this is real and not cosmetic: `desc_act` interacts with
  tensor-parallel row splitting badly enough that vLLM carries an `is_k_full` flag for it
  (`auto_gptq.py:516`), and the config constructor silently forces `desc_act = False` when
  `group_size == -1` because with one group per channel the ordering is a no-op
  (`auto_gptq.py:118`).
- **AWQ** `[C]` (2306.00978) makes the cheaper observation that you do not need the
  Hessian, only its diagonal: find the ~1% of input channels with the largest activation
  magnitude and *scale them up* before quantizing (then scale the activations down by the
  same factor, using SmoothQuant's identity), so those channels land further from the
  rounding boundary. No backprop, no reconstruction, one calibration pass.

> **Systems bridge.** This is importance-weighted lossy compression — the same idea as a
> perceptual codec allocating bits by psychoacoustic masking rather than uniformly.
>
> **Where it breaks.** A perceptual model is fixed and known; `H` is estimated from a
> calibration set and is a property of the *input distribution*. Change the workload — a
> different language, a different code style, a longer context — and the weighting you
> optimized against is wrong, with no signal that it is wrong. This is the same
> prematurity flaw `kv-eviction-policies.md` §4 identifies in eviction policies:
> **an irreversible decision made against a proxy for a distribution that has not
> arrived.** Weight quantization and KV eviction are the same failure mode at different
> timescales, which is why `[C]` 2607.08032 (Jul 2026) can unify eviction, prompt
> compression, recurrent-state bounding and agent-memory consolidation as one
> rate-distortion problem.

### 3.6 KV quantization: same quantizer, and the axis you cannot stream along

The KV cache is a `[tokens × channels]` matrix per layer per head. Choosing a group means
choosing a direction across that matrix, and there are two:

- **Per-token** (group along channels, within one token): one scale per token per head.
  Streams perfectly — the token arrives, you compute its scale, you store it.
- **Per-channel** (group along tokens, within one channel): one scale per channel, over a
  run of tokens. Does **not** stream — you cannot know a channel's max until you have seen
  all the tokens in the group.

`[C]` KIVI's empirical finding (2402.02750, Feb 2024), which every subsequent quantizer
inherits: **keys carry large outliers that are consistent per channel; values do not.**
Therefore quantize keys per-channel and values per-token. `[C]` KVQuant (2401.18079) adds
two refinements — quantize keys *before* RoPE, because the rotation smears a channel's
statistics across pairs of channels and destroys the very structure you were exploiting,
and isolate outliers per-vector.

Now the problem, stated as arithmetic. Per-channel key quantization needs a *block of
tokens*. The decode loop produces *one token*. So the implementation must hold a
full-precision staging buffer of the most recent `r` tokens and flush it into quantized
per-channel blocks when it fills. That is precisely what the reference implementation
does: `cache_utils.py:690`, `residual_length: int = 128`.

> **Systems bridge — the good one.** You want columnar statistics over a row-oriented
> write path. That is Parquet row groups, or an LSM memtable: buffer rows, flush a column
> chunk, compute per-column statistics at flush time. The staging buffer size is a
> classic write-amplification-versus-metadata knob.
>
> **Where it breaks, three places.**
> 1. **The flush is lossy and one-way.** An LSM flush preserves bytes; this one destroys
>    them. You cannot re-derive the pre-flush values, and there is no compaction pass that
>    could improve them later with better statistics.
> 2. **`residual_length` is an accuracy knob wearing a latency knob's clothes.** Bigger
>    `r` means more full-precision recent context — which is exactly the context the model
>    attends to most — so the quality curve against `r` is not the throughput curve
>    against `r`, and nobody reports both.
> 3. **The reference implementation does not implement the paper.** Read the docstring at
>    `cache_utils.py:672`: it cites KIVI by name and then states that quantization "is
>    done per-channel with a set `q_group_size` for both Keys and Values, **in contrast to
>    what was described in the paper**," with `axis_key = axis_value = 0`. The asymmetry
>    that *is* KIVI's contribution is absent from the class that names it. If you
>    benchmark "KIVI in transformers" you are benchmarking something else — the same class
>    of harness error `kv-cache-mechanics.md` §4.3 warns about, one level down.

### 3.7 The scale tensor is part of the cache, and it does not amortize

For weights, scale overhead is divided across the whole model and across every request
forever. For KV, it is paid **per token, per head, per layer, per sequence**. Read
vLLM's accounting directly (`kv_cache_interface.py:185`):

```python
if self.kv_quant_mode.is_per_token_head:
    unpadded += 2 * self.block_size * self.num_kv_heads * get_dtype_size(torch.float32)
```

Two fp32 scales (one for K, one for V) per token per KV head. Per token per layer that is
`2 · n_kv · 4` bytes. For Laguna-S (`n_kv = 8`, `d_h = 128`, `L = 48`) `[M]`:

| KV format | payload / token (all 48 layers) | scale / token | total | overhead | vs bf16 |
|---|---|---|---|---|---|
| bf16 | 192.0 KiB | 0 | **192.0 KiB** | — | 1.00× |
| `q8_0` (ggml, 8.5 bits) | 102.0 KiB | in-block | **102.0 KiB** | 6.25% | 1.88× |
| fp8 per-tensor | 96.0 KiB | ~0 | **96.0 KiB** | ~0% | 2.00× |
| int8 per-token-head | 96.0 KiB | 3.0 KiB | **99.0 KiB** | **3.13%** | 1.94× |
| int4 per-token-head | 48.0 KiB | 3.0 KiB | **51.0 KiB** | **6.25%** | 3.76× |
| int2 (hypothetical, same scales) | 24.0 KiB | 3.0 KiB | **27.0 KiB** | **12.5%** | 7.11× |

Arithmetic over `[M]` inputs (`ASSUMPTIONS.md → kv-per-token-laguna`, and the vLLM formula
above). The last column is the number people quote; the overhead column is the number they
omit. **At 2 bits the metadata is an eighth of the payload**, which is large enough to
reorder a leaderboard — and no sub-4-bit compression-ratio claim this lab has read
includes it.

Note also what the ggml column shows: block quantization folds its scale *into* the block,
so `q8_0` at 8.5 bits/element is *already* the honest number, while "int8 with per-token
scales" needs the extra line. Two different, equally valid accounting conventions, and
comparing across them without normalizing is a mistake that is easy to make and invisible
in a results table.

**One more structural fact, read from the source.** The reference engine allows exactly
nine KV cache types (`common/arg.cpp:301`): `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl,
q5_0, q5_1`. There is **no fp8**, and there are **no k-quants** — every quantized entry on
that list is a simple 32-element block, and the 256-element super-block formats that
dominate weight storage are absent. `[A]` Medium confidence the reason is that a
super-block's two-level scale fit is a batch operation over 256 elements and the cache
write path appends one token's 128-wide head row at a time, so a super-block would either
straddle tokens (unappendable) or need re-fitting on every write. Cheapest test that would
move it: read the cache write path in `llama-kv-cache.cpp` and check whether it calls
`ggml_quantize_chunk` per row. Either way the fact stands: **the KV cache's format menu is
a strict subset of the weights' format menu, and it is a different subset than you would
guess.**

### 3.8 The traffic accounting on hardware with no low-precision matmul

This is where our machine stops being an ordinary instrument. `[M]` `torch._scaled_mm`
raises `RuntimeError: only supported on CUDA devices with compute capability >= 9.0 or
8.9, or ROCm MI300+` on gfx1151 (torch `2.12.0a0+rocm7.13.0a20260313`; recorded in
`research/memory/kv-cache-mechanics.md` and `research/notes/inference-and-quantization.md`,
and re-confirmable by Exercise B). There is also no Marlin path: vLLM gates its fast
low-bit kernels on `current_platform.is_cuda()` (`auto_awq.py:312`) and its FP8 fallback
for hardware without FP8 units is *also* Marlin (`fp8.py:287`).

So on this box, "quantized weights" means: store few bytes, **materialize a bf16 copy**,
then run an ordinary bf16 GEMM. Count the traffic for one weight matrix of `K·N` elements
at batch `M`:

```
bf16 path :  read 2·K·N                                      = 2   B/element
int4 path :  read 0.5·K·N (payload)  +  ~0.02 (scales)
             + write 2·K·N (the bf16 temporary)
             + read 2·K·N (the GEMM reads the temporary)      ≈ 4.5 B/element

predicted ratio at batch 1  =  4.5 / 2  =  2.25×  SLOWER
```

That is the same shape of result as `kv-cache-mechanics.md` §3.7 got for fp8 KV (predicted
2.5×, measured 2.9–3.1× `[M]`), and for the same reason: **the dequantized temporary is
traffic, and unfused dequantization is a read-modify-write over the entire tensor.**

Now the part that makes it a teaching point rather than a complaint. That penalty is
*amortized by batch*, because the dequantize runs once per weight matrix per forward pass
regardless of `M`, while the useful FLOPs scale with `M`. So:

```
                    (0.5 + 2 + 2)·K·N            4.5
slowdown(M)  ≈  ─────────────────────────  →  ─────── at M=1,  → 1 as M grows large
                 2·K·N + (dequant fixed)         2
```

**Unfused weight quantization hurts most exactly where it is supposed to help most: at
batch-1 decode.** That is a sharp, falsifiable prediction and it is Exercise C.

`[M]` **Measured, and the direction was right while the magnitude was wrong by 5×.**
Exercise C, `K = N = 8192`, group 128, bf16, two independent reps in isolated fresh
subprocesses per cell, median of 5 × 20 iterations per cell, gfx1151, hipBLASLt
configured, 2026-07-26:

| `M` | bf16 GEMM | dequant only | int4 path | **slowdown** | dequant share | bf16 weight-read |
|---|---|---|---|---|---|---|
| 1 | 0.659 / 0.658 ms | 8.077 / 8.095 ms | 8.644 / 8.847 ms | **13.12 / 13.45×** | 93% / 92% | 203.7 / 204.1 GB/s |
| 8 | 0.650 / 0.661 ms | 8.050 / 8.067 ms | 8.853 / 8.842 ms | **13.61 / 13.39×** | 91% / 91% | 206.3 / 203.2 GB/s |
| 512 | 3.700 / 3.659 ms | 8.105 / 8.139 ms | 10.978 / 11.128 ms | **2.97 / 3.04×** | 74% / 73% | (compute-bound) |

Three things fall out, and the third is the one worth carrying.

**The amortization argument held exactly.** `dequant only` is **8.05–8.14 ms at every
batch size** — flat to within 1%, across six measurements in six processes. It is a fixed
cost per weight matrix per forward pass, precisely as the model says, and the slowdown
collapses from 13× to 3× as the bf16 side grows. So the *shape* of `slowdown(M)` is right.

**The magnitude was 5.8× worse than predicted, and the reason is eager mode.** The traffic
model assumed one unpack producing one temporary — 4.5 B/element. The obvious PyTorch
expression materializes **four**: `(q & mask).to(bf16)`, `(q >> 4).to(bf16)`,
`torch.stack(...).reshape(...)`, and the scale multiply. Counting them gives roughly
`(33.5 + 67) + (33.5 + 67) + (134 + 134) + (134 + 134) ≈ 737 MB` per dequantize against a
134 MB minimum, i.e. **5.5× the traffic of the model** — which lands almost exactly on the
5.8× discrepancy. The measured dequantize achieves ~91 GB/s effective, versus 204 GB/s for
the contiguous GEMM read, so it is bandwidth-bound on a badly-shaped access pattern rather
than launch-bound. **The 2.25× figure is the floor for a single fused unpack kernel;
13× is the price of writing the unpack as an expression.** Same lesson as
`kv-cache-mechanics.md` §3.4's `torch.cat`: in this domain the data structure and the
kernel boundary *are* the performance model, and the algebra only bounds them.

**`[M]` And a genuinely new number for the lab, which was not what this exercise was
for.** A batch-1 bf16 GEMV over a contiguous 134 MB weight matrix reaches
**203–206 GB/s** of weight-read bandwidth — *at or slightly above* the 199.9 GB/s
device-to-device copy figure `[M]`, and **~1.36× the ~150 GB/s that decode-shaped
attention reads achieve** (`kv-cache-mechanics.md` Exercise B `[M]`). Both are batch-1,
both are memory-bound, both read a large tensor once. So the attention deficit is not a
general property of small-`M` matmuls on this machine; it localizes to attention's
**access pattern** — a strided gather across `n_kv` head slices feeding an `M = 6` matmul
— rather than to GEMV shape. That partially answers open question 3 of
`kv-cache-mechanics.md` §8 (the attention-kernel ridge point): the *weights* half of the
roofline denominator is ~204 GB/s and the *attention* half is ~150, and a single-number
ridge is wrong for both. Prediction 4 in Exercise C guessed 130–160 GB/s and was
falsified in the good direction.

The honest scope: a real serving stack fuses the dequantize into the GEMM prologue and
never writes any temporary, which removes essentially all of the 737 MB and makes int4
weights a genuine ~3.9× bandwidth win at batch 1. **We cannot measure that path** — there
is no Marlin, no CUTLASS low-bit kernel, and no `_scaled_mm` on ROCm gfx1151 `[M]`. So the
13× is a true statement about *plain PyTorch on this box* and a false statement about
weight quantization. Label it that way every time, or the number will be quoted back at
you as an argument against a technique it does not measure.

### 3.9 What halving `b` does to arithmetic intensity, and why it is not enough

From `kv-cache-mechanics.md`, decode attention intensity is `AI = 2G/B_e` where `B_e` is
bytes per element. Halving `B_e` doubles `AI` with no interaction term against GQA, MLA,
windowing, or eviction — it really is the cleanest knob in the system. But run the numbers
against our `[M]` ridge of ≈105 FLOP/byte:

| KV dtype | `AI` on Laguna's global layers (`G = 6`) | % of the 105 ridge |
|---|---|---|
| bf16 | 6 | 5.7% |
| fp8 / int8 | 12 | 11.4% |
| int4 | 24 | 22.9% |
| int2 | 48 | 45.7% |

Even a **2-bit** KV cache leaves decode attention below half the ridge point. Quantization
does not move decode from bandwidth-bound to compute-bound; it moves it from *deeply*
bandwidth-bound to *somewhat less* bandwidth-bound. Anyone promising otherwise is quoting
a throughput number that came from a batching change or a kernel fusion, not from the
dtype. (And per `kv-cache-mechanics.md` Exercise B, the 105 itself is built on the wrong
denominator for attention — the measured decode-attention read rate is ~150 GB/s, not
199.9 — so the real percentages are worse.)

---

## 4. Why this matters for Proteus and Mnemosyne

### 4.1 The half-bit tax decides whether the reference model fits

Laguna-S: 48 layers, `hidden = 3072`, `moe_intermediate = 1024`, 256 experts, vocab
100,352, `layer_types` 12 full + 36 sliding, `mlp_only_layers = [0]` `[M]` (read from
`research/reference/models/laguna-s/config.json` at revision `b0a9fd7c850e`).

Parameter decomposition, arithmetic over that read:

```
routed experts   47 sparse layers × 256 experts × 3 mats × 3072 × 1024   = 113.5e9
shared experts   47 × 3 × 3072 × 1024                                    =   0.44e9
dense layer 0    3 × 3072 × 12288                                        =   0.11e9
attention        12×(2·3072·6144 + 2·3072·1024) + 36×(2·3072·9216 + …)    ≈   2.8e9
embed + lm_head  2 × 100352 × 3072                                       =   0.62e9
                                                                    total ≈ 117.5e9
```

Consistent with the advertised 118B. **Experts are ~96.6% of the parameters**, which is
why every shipped MoE quantization recipe quantizes experts and nothing else — `[M]` our
local `gpt-oss-20b/config.json` does exactly this, with `modules_to_not_convert` excluding
attention, the router, embeddings and `lm_head`, and all 96 `*_blocks`/`*_scales` tensor
pairs belonging to `mlp.experts.{gate_up,down}_proj`.

Now the capacity question, against our `[M]` measured **≥62 GiB fast tier at ~200 GB/s**
(`ASSUMPTIONS.md → gpu-fast-tier-size`):

| Recipe | Expert bytes | Everything-else bytes | Total | vs 62 GiB fast tier |
|---|---|---|---|---|
| all bf16 | 227.9 GB | 7.1 GB | 235.0 GB = 218.9 GiB | 3.5× over — does not fit the *machine* |
| all `q8_0` (8.5b) | 121.0 GB | 3.8 GB | 124.8 GB = 116.2 GiB | over the whole 128 GB pool with the OS in it |
| experts `q4_K`/NVFP4 (4.5b) | 64.1 GB | 7.1 GB | 71.2 GB = **66.3 GiB** | **over** |
| experts MXFP4 (4.25b) | 60.5 GB | 7.1 GB | 67.6 GB = **62.9 GiB** | **at the edge** |
| experts MXFP4 + rest fp8 | 60.5 GB | 3.5 GB | 64.0 GB = **59.6 GiB** | **fits, ~2 GiB headroom** |

Arithmetic over `[M]` inputs. Sit with the third and fourth rows: **a quarter of a bit per
weight is the difference between the reference model's weights living in the measured fast
tier and not.** This is exactly why §3.3 exists, and it is the most concrete possible
answer to "why does a memory-systems lab care about number formats."

Two caveats, stated because the table is seductive. The 62 GiB figure is a *floor* — the
sweep hit the ≥32 GiB single-tensor fault, not a bandwidth knee `[M]` — so the fourth row
may well fit in practice; we do not know because the upper edge is unmeasured. And we
cannot run Laguna-S today regardless: there is no GGUF, no engine configured, and 118B is
outside every other budget we have. This is a capacity-planning calculation, not an
experiment. Say so in any write-up.

### 4.2 `kv_dtype` is a Mnemosyne field, and quantization and eviction share one budget

`kv-cache-mechanics.md` §4.5 already argues that `kv_dtype` belongs on the config surface.
This module adds the reason it belongs to **Mnemosyne** rather than Proteus: the format is
a property of the inference path, not the architecture. The reference engine proves it
twice over — llama.cpp takes `-ctk`/`-ctv` as *runtime flags* (`arg.cpp:2190`) with nothing
in the model config mentioning them, and vLLM carries `kv_quant_mode` as a field of the
*cache spec* (`kv_cache_interface.py:180`), not of the model.

The stronger claim, and the one worth designing for: **quantization and eviction are two
ways of spending the same bit budget, and the literature has started to say so.** `[C]`
RDKV (2605.08317, May 2026) allocates bits between eviction and quantization under one
rate-distortion objective; `[C]` EvicPress (2512.14946) does joint compression and
eviction; `[C]` 2607.08032 (Jul 2026) unifies the whole family. Concretely: at a fixed
byte budget you can keep `N` tokens at 16 bits or `2N` at 8 or `4N` at 4, and nobody knows
where the optimum sits for a given task. That is a clean, cheap, matched-budget ablation
at our scale and it is the single most obvious experiment this module implies.

Design consequence for Mnemosyne's interface (extending the three plug points in
`research/memory/kv-compression-and-eviction.md` §6): the write-time admission hook must be
allowed to return *a precision* as well as a keep/drop decision. A `bool` return type
forecloses the entire joint budget. That is a two-minute interface decision now and an
expensive refactor later.

### 4.3 The cost model must count the scales, and must count them per token

`kv-cache-mechanics.md` §4.1 specifies Mnemosyne's cost model as
`list[LayerCacheSpec] → residency_bytes(T) / read_bytes_per_step(T) /
maintenance_bytes_per_step(T)`. This module adds one required field to that spec —
the quantization mode — and one required behaviour: **`residency_bytes` must include the
scale tensor, and the scale tensor's size must be a function of `T`, not a constant.**
vLLM's `unpadded_page_size_bytes` is the reference implementation of exactly that
(`kv_cache_interface.py:185`), and its int4 mode is implemented by *halving `head_dim`*
against a 1-byte dtype (`:208`) rather than by introducing a sub-byte dtype — a packing
convention worth copying, because it keeps every downstream stride calculation integral.

### 4.4 The confound, stated plainly, because it invalidates the obvious experiment

The obvious experiment is "measure quality loss versus bit width on our own model." Today
we cannot cleanly do it, and the reason is not the quantizer:

- `[M]` **bf16 numerics on gfx1151 are unproven** (`ASSUMPTIONS.md →
  bf16-numerics-unproven`; five critical bf16 bugs documented `[C]` for this silicon). Our
  *reference* precision is under suspicion, so a bf16-vs-int4 comparison measures
  `(quantization error) ⊕ (bf16 error) ⊖ (bf16 error)` only if the two errors are
  independent, and nobody has shown they are.
- `[M]` **hipBLASLt configuration changes long-reduction bf16 relative error by 2.8×**
  (2.01e-3 configured vs 5.60e-3 unset at N = 2²⁰, 3 seeds, fresh processes;
  `ASSUMPTIONS.md → hipblaslt-config`). It is a numerics control, not a throughput knob.
  Any quantization-error run must record it.
- `[M]` **`allow_bf16_reduced_precision_reduction` is inert on this stack** — toggling it
  changes the result by exactly zero bits (`ASSUMPTIONS.md →
  bf16-reduced-precision-knob-works`, **refuted**). Do not reach for it as a control; it
  is an API that reports a capability it does not deliver.
- `[M]` **SDPA retains the score matrix by default** (147.2 vs 6.6 bytes/T² with
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`; `ASSUMPTIONS.md →
  sdpa-is-memory-efficient`), and the flag that fixes it is itself a numerics change. So
  the attention path you measure quantization error *through* has two settings and we have
  not compared them.

**The rule this implies, and it should go in the pre-registration of every quantization
arm we run:** compare the quantized arm against an **fp64 or CPU reference**, never
against a bf16 GPU reference, and record `hipblaslt_libpath_set`,
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL`, the wheel version and the driver version in the
run header. Exercise A is built that way on purpose: it computes error against fp32/fp64
on CPU, which is the only reference on this machine currently entitled to be called
correct.

### 4.5 Report error at three points, not one

The attribution gap this lab claims as its edge applies directly. A quantization result
reported as one accuracy delta is unattributable. Report:

1. **Tensor error** — `‖W − Ŵ‖ / ‖W‖`, and the outlier ratio `m/σ` that produced it.
   Free; one pass over the artifact.
2. **Layer output error** — `‖xW − xŴ‖ / ‖xW‖` on real activations. This is the quantity
   §3.5 says actually matters, and the gap between (1) and (2) tells you whether your
   quantizer is Hessian-aware in effect if not in name.
3. **Behavioural probe** — not perplexity. `[C]` 2606.09864 (Jun 2026) reports Mistral-7B
   losing **15.2% of its refusals at 1.03× perplexity** across eleven instruction-tuned
   models and 1,894 prompts under KV quantization, with no universal safe bit-width and a
   stated low-dimensional-subspace mechanism. Perplexity integrates over all directions;
   the failure lives in a few. Outcome metric fine, mechanism broken — the lab's own
   diagnosis of the literature, in someone else's artifact.

---

## 5. Read the code

All paths relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. Line numbers are pinned to the revisions in
`PROVENANCE.md`.

### 5.1 The formats, as C structs with size assertions

The single highest-value read in this module. Twenty minutes with this file and you will
never again believe a nominal bit count.

| Where | What to look at, and why |
|---|---|
| `architecture/llama-cpp-laguna/ggml/src/ggml-common.h:90` | `#define K_SCALE_SIZE 12` — twelve bytes of *quantized* scales per 256-element super-block. The scale hierarchy exists because §3.4 says granularity is the lever and metadata is the price. |
| `architecture/llama-cpp-laguna/ggml/src/ggml-common.h:215` | `block_mxfp4` — one `uint8_t e` (an E8M0 power-of-two exponent) plus 16 bytes of packed E2M1. 17 bytes, 32 values, 4.25 bits. The OCP microscaling format `[C]` (2310.10537) in nine lines. |
| `architecture/llama-cpp-laguna/ggml/src/ggml-common.h:223` | `block_nvfp4` — **four** UE4M3 sub-block scales (one per 16 values) plus 32 bytes of E2M1. Put it next to `block_mxfp4` and the entire MXFP4-vs-NVFP4 argument is visible as a struct diff: finer groups, richer scale, +0.25 bits. |
| `architecture/llama-cpp-laguna/ggml/src/ggml-common.h:327` | `block_q4_K` — two fp16 super-scales, 12 bytes of 6-bit sub-scales, 128 bytes of nibbles. Read the comment three lines above it: the source states "Effectively 4.5 bits per weight" itself. |
| `architecture/llama-cpp-laguna/ggml/src/ggml-common.h:256` | The `static_assert` under `block_q8_0`. Every block type has one. This is how a wire format is supposed to be defended, and it is worth stealing for any on-disk structure Mnemosyne ever grows. |
| `architecture/llama-cpp-laguna/ggml/src/ggml-common.h:181` | `block_q1_0` — 1.125 bits/weight, 128 values per fp16 scale. The far end of the tradeoff, shipping in the reference engine. |

### 5.2 The KV cache's format menu is a different menu

| Where | What to look at, and why |
|---|---|
| `architecture/llama-cpp-laguna/common/arg.cpp:301` | `const std::vector<ggml_type> kv_cache_types` — nine entries: `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`. **No fp8. No k-quants.** Every quantized entry is a 32-element block. §3.7 argues why. |
| `architecture/llama-cpp-laguna/common/arg.cpp:2190` | `-ctk / --cache-type-k` — the format is a *command-line flag*, defaulting per build, with nothing in the model config. The cache format is a property of the inference path, not of the model. That sentence is the reason `kv_dtype` belongs to Mnemosyne. |
| `architecture/llama-cpp-laguna/src/llama-kv-cache.cpp:319` | `attn_rot_k = !attn_rot_disable && … && ggml_is_quantized(type_k) && hparams.n_embd_head_k() % 64 == 0` — the rotation lineage (§3.4) as a four-term runtime predicate. Turning on a quantized cache silently turns on a Hadamard transform of K. |
| `architecture/llama-cpp-laguna/src/llama-kv-cache.cpp:344` | The precompute loop that builds Hadamard matrices for every power-of-two size from 64 up, at cache construction, in host memory. Cheap, one-time, and completely invisible from outside. |
| `architecture/llama-cpp-laguna/src/llama-kv-cache.cpp:313` | `getenv("LLAMA_ATTN_ROT_DISABLE")` — the only way to turn it off, documented nowhere. If you A/B a quantized KV cache in this engine without setting it, you are A/B-ing quantization *and* rotation together. |

### 5.3 vLLM's KV quantization: the taxonomy, in an enum

| Where | What to look at, and why |
|---|---|
| `memory/vllm/vllm/v1/kv_cache_interface.py:33` | `class KVQuantMode(IntEnum)` — the whole design space as six values: `NONE, FP8_PER_TENSOR, INT8_PER_TOKEN_HEAD, FP8_PER_TOKEN_HEAD, INT4_PER_TOKEN_HEAD, NVFP4`. Note what is *not* there: no per-channel key mode. The production menu does not include KIVI's actual recommendation. |
| `memory/vllm/vllm/v1/kv_cache_interface.py:44` | `INT4_PER_TOKEN_HEAD = 4  # packed 2×int4/byte, RHT + asymmetric zp` — a one-line comment containing three separate ideas from §3: sub-byte packing, randomized Hadamard rotation, asymmetric zero-point. |
| `memory/vllm/vllm/v1/kv_cache_interface.py:185` | `unpadded_page_size_bytes` — §3.7's scale accounting, in production. Two fp32 scales per token per KV head, explicitly budgeted out of the KV allocation because "the memory is carved from the raw KV cache allocation." |
| `memory/vllm/vllm/v1/kv_cache_interface.py:209` | `head_dim = self.head_size // 2` for int4 (the branch opens at `:208`) — sub-byte storage implemented by halving the logical head dimension against a 1-byte dtype, so nothing downstream needs a sub-byte stride. |
| `memory/vllm/vllm/model_executor/layers/quantization/kv_cache.py:42` | `class BaseKVCacheMethod` — read the docstring: quantize on the way in, dequantize on the way out. That is the whole contract, and it is where §3.8's traffic goes. |
| `memory/vllm/vllm/model_executor/layers/quantization/kv_cache.py:65` | Four scales created per attention layer: `q_scale`, `k_scale`, `v_scale`, **`prob_scale`** — the last quantizes `softmax(QKᵀ)`. §2.3. This is the parameter nobody's compression ratio mentions. |
| `memory/vllm/vllm/model_executor/layers/quantization/kv_cache.py:130` | `"Only support per-tensor scaling factor for fp8 KV cache"` (the `raise` guard opens at `:128`) — the fp8 path is per-*tensor*, the coarsest possible group. §3.2 says that maximizes `m_G`. The production default is the least accurate grouping available. |
| `memory/vllm/vllm/model_executor/layers/quantization/kv_cache.py:150` | The warning fired when both scales load as 1.0: an uncalibrated fp8 cache runs silently and looks identical to a calibrated one. This is the exact failure mode Break 1 predicts. |
| `memory/vllm/vllm/model_executor/layers/quantization/kv_cache.py:85` | For per-token-head modes, checkpoint scales are *discarded* and set to 1.0 because the kernel computes them dynamically. So "the checkpoint has KV scales" tells you nothing about whether they are used. |

### 5.4 FP8 is not one number system — and this is an AMD-specific trap

| Where | What to look at, and why |
|---|---|
| `memory/vllm/vllm/model_executor/layers/quantization/utils/w8a8_utils.py:110` | `def normalize_e4m3fn_to_e4m3fnuz` — a whole function whose job is to convert between two eight-bit float formats that occupy the same eight bits. |
| `memory/vllm/vllm/model_executor/layers/quantization/utils/w8a8_utils.py:116` | `# The bits pattern 10000000(-128) represents zero in e4m3fn but NaN in e4m3fnuz.` The same byte is a valid zero in one system and a NaN in the other. A byte-for-byte copy of a checkpoint between the two produces NaNs. |
| `memory/vllm/vllm/model_executor/layers/quantization/utils/w8a8_utils.py:128` | `weight_scale = weight_scale * 2.0` — for identical bits, the fnuz value is **half** the fn value, so every scale must double. Get this wrong and your model runs, produces plausible tokens, and is wrong by 2× on every quantized tensor. |
| `memory/vllm/vllm/platforms/rocm.py:890` | `return "gfx94" in _GCN_ARCH` — the entire detection rule, one substring test, inside `is_fp8_fnuz` at `:888`. **gfx1151 is not gfx94x**, so on our machine the OCP formats (`e4m3fn`/`e5m2`) are the correct ones and the fnuz variants, which torch will happily allocate for us, are a trap. |
| `memory/vllm/vllm/platforms/interface.py:1081` | "AMD's MI300 and MI325 have native hardware support for FNUZ. All other hardware has converged on the OCP FP8 standard." A one-sentence summary of a format war, in a docstring. |

### 5.5 Weight quantization: the wire formats have opinions

| Where | What to look at, and why |
|---|---|
| `memory/vllm/vllm/model_executor/layers/quantization/auto_awq.py:77` | `_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]` — AWQ packs nibbles into int32 in a non-sequential order, for kernel convenience, and every consumer must undo it. A wire format with an interleave, exactly like a disk layout optimised for a particular read pattern. |
| `memory/vllm/vllm/model_executor/layers/quantization/auto_awq.py:193` | `self.pack_factor = 32 // weight_bits  # packed into int32` — sub-byte values live inside int32 words. This is why 3-bit and 5-bit formats are rare: they do not divide 32. |
| `memory/vllm/vllm/model_executor/layers/quantization/auto_awq.py:312` | `and current_platform.is_cuda()` — the Marlin fast-kernel path is CUDA-gated. On ROCm the code falls through to "unoptimized AWQ kernels" with a `warning_once`. **This is the line that makes §3.8 true for us.** |
| `memory/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:110` | `desc_act: bool` — GPTQ's activation-order option, promoted to a first-class config field. §3.5. |
| `memory/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:118` | `if desc_act and group_size == -1: desc_act = False` — with one group per channel, ordering is a no-op. A three-line comment that tells you exactly what `desc_act` is for. |
| `memory/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:373` | `scales_and_zp_size = input_size // group_size` — §3.3's denominator, in allocation code. |
| `memory/vllm/vllm/model_executor/layers/quantization/fp8.py:273` | The `Limitations:` docstring — "Only support `float8_e4m3fn` … due to the limitation of `torch._scaled_mm`", with a permalink into the ATen source. The format menu is set by one kernel. |
| `memory/vllm/vllm/model_executor/layers/quantization/fp8.py:287` | `# For GPUs that lack FP8 hardware support, we can leverage the Marlin kernel` — and Marlin is CUDA-only. Both escape hatches close on the same platform check. |

### 5.6 The reference KV quantizer, and why it is not the paper

| Where | What to look at, and why |
|---|---|
| `architecture/transformers/src/transformers/cache_utils.py:672` | `class QuantizedLayer(DynamicLayer)` — read the docstring twice. It cites KIVI `[C]` (2402.02750) and then states the implementation quantizes per-channel for **both** K and V, "in contrast to what was described in the paper." The asymmetry that is KIVI's contribution is not implemented by the class that names it. |
| `architecture/transformers/src/transformers/cache_utils.py:690` | `residual_length: int = 128` — the staging buffer from §3.6. An accuracy knob presented as a capacity knob. |
| `architecture/transformers/src/transformers/cache_utils.py:722` | `dequant_keys = self._dequantize(self._quantized_keys)` — **the entire cache is dequantized on every single decode step**, then `:724` concatenates three tensors to rebuild it. This is §3.8's traffic multiplied by `torch.cat`'s `O(T²)` maintenance defect from `kv-cache-mechanics.md` §3.4. Benchmarking KV quantization here measures the harness. |
| `architecture/transformers/src/transformers/cache_utils.py:726` | The flush condition — quantize only when the residual buffer is full. The LSM memtable, in four lines. |

### 5.7 One more, because layout and quantization interact

| Where | What to look at, and why |
|---|---|
| `memory/flashinfer/flashinfer/decode.py:1982` | A bare `print("[WARNING] NVFP4 KV cache with NHD layout will be converted to HND, incurring extra transpose and contiguous copy overhead…")` — on **every call**, to stdout, not through logging. |
| `memory/flashinfer/flashinfer/decode.py:1989` | `key_block_scales = key_block_scales.transpose(-3, -2).contiguous()` — and the *scale tensor* gets copied too. A quantized cache is two tensors, and both have to agree with the kernel's layout. Any Mnemosyne interface that models a KV page as one buffer cannot express NVFP4. |

---

## 6. Exercises

> **Declared defect, read this before you start.** **Exercises B and C were run on the Z13
> before shipping and carry `[M]` reference tables. Exercise A was not**, and ships with
> predictions only — the run that was meant to produce it discovered that the model
> weights are Git LFS pointer stubs (see A's preamble, which is now the finding), and the
> session's process-spawn layer was too degraded to iterate further. That is a defect, and
> the house rule is to declare it rather than to fill a table with plausible numbers.
>
> Between them B and C produced four `[M]` results, **two of which falsified predictions
> written in this file before the run** — the batch-1 int4 slowdown (predicted 2.25×,
> measured 13.1–13.6×) and the batch-1 weight-read bandwidth (predicted 130–160 GB/s,
> measured 203–206). Both corrections are folded back into §3.8 rather than quietly
> deleted, and both changed what the module concludes.
>
> Runtime figures for B and C are measured; A's is an estimate and is marked as such.

Activate first, in PowerShell, dot-sourced so the variables survive:

```powershell
. .\scripts\activate-lab.ps1
```

**Standing hardware caveats** (`ASSUMPTIONS.md`): keep every individual tensor under
31 GiB — single tensors ≥32 GiB **hang silently at 0% CPU** `[M]`; bf16 numerics are
unproven `[M]`, so *accuracy* claims from these exercises are provisional in a way that
*byte-count* claims are not; `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` is deliberately
**off** in `activate-lab.ps1` and these exercises assume it is off. Record
`HIPBLASLT_TENSILE_LIBPATH` state in every run — §4.4.

Write scratch scripts under `notebook/`. Exercises A and B are Hardware Validation Gate
candidates and should migrate into the rig with tests on reuse.

---

### Exercise A — group size versus outlier ratio: the §3.2 inequality, measured

> **`[M]` Blocker discovered while preparing this module, and you need it before you
> start.** The intent was to run this on the reference model's real weights. **They are
> not on disk.** Probed 2026-07-26 in a fresh process: for all six model directories under
> `research/reference/models/` — `gpt-oss-20b`, `kimi-linear-model`, `laguna-s`,
> `laguna-xs`, `nemotron-nano`, `qwen3-next` — the first `*.safetensors` shard is
> **135 bytes** and begins with the hex `76657273696f6e2068747470` = ASCII
> `"version http"`. Those are **Git LFS pointer files**, not tensors. The `config.json`
> and `model.safetensors.index.json` files *are* real, which is why every `[M]` config
> read in this curriculum is sound and why nobody had noticed: the whole repo has been
> reading metadata.
>
> Consequences, in order of importance. (1) `scripts/fetch_reference.sh` needs an LFS
> step, or the models need explicit exclusion from the manifest — either way it is a
> `BACKLOG.md` item, not something to work around here. (2) Any future exercise that
> touches weights must probe first; the failure mode is a `MemoryError` from
> `struct.unpack` reading `"version "` as a little-endian `uint64` header length
> (≈2.3×10¹⁸ bytes), which is an *extremely* confusing way to learn that your data is
> absent. (3) This exercise is therefore built on synthetic weights with a **controlled**
> outlier ratio, which is a fair trade: it tests §3.2's inequality directly, with the
> confound removed, and it gives you a closed-form prediction to check against. Step 0
> re-probes for real weights so that the day they arrive you get both.

**Goal:** verify that quantization error is set by the group maximum, not by the bit count
alone, and put a number on the exchange rate between the two.

**Hardware:** CPU only, deliberately. Nothing here needs the GPU, and using it would put
unproven bf16 numerics (§4.4) inside a measurement whose whole point is precision.
**Runtime:** ~3 minutes (estimate) at the shapes given.

```python
"""Group size vs outlier ratio. CPU only, fp32 reference, deterministic seed.
Step 0 probes for real weights; the sweep runs on synthetic ones either way."""
import json, struct
from pathlib import Path
import torch

REF = Path("research/reference/models")

# ---- step 0: do we actually have weights? -------------------------------
for model in sorted(p for p in REF.iterdir() if p.is_dir()):
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        continue
    head, size = shards[0].open("rb").read(16), shards[0].stat().st_size
    print(f"{model.name:20} {len(shards):3d} shards  first={size:>14,} B  "
          f"{'LFS POINTER' if head[:7] == b'version' else 'real tensors'}")

# ---- the quantizers (identical to the ones you would run on real weights)
def q_int(x, bits, group):
    """group=None -> per tensor; -1 -> per output row; g -> g inputs share a scale.
    Returns (dequantised tensor, bytes of fp16 scale storage)."""
    qmax = 2 ** (bits - 1) - 1
    if group is None:
        s = x.abs().max().clamp_min(1e-12) / qmax; n = 1
        deq = torch.round(x / s).clamp(-qmax - 1, qmax) * s
    elif group == -1:
        s = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax; n = x.shape[0]
        deq = torch.round(x / s).clamp(-qmax - 1, qmax) * s
    else:
        r, c = x.shape
        xg = x.reshape(r, c // group, group)
        s = xg.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax; n = r * (c // group)
        deq = (torch.round(xg / s).clamp(-qmax - 1, qmax) * s).reshape(r, c)
    return deq, n * 2

def q_fp8(x, dt):
    s = (x.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-12)
    return ((x / s).to(dt).to(torch.float32)) * s, x.shape[0] * 2

def relerr(x, d):
    return float(torch.linalg.vector_norm(x - d) / torch.linalg.vector_norm(x))

# ---- the sweep ----------------------------------------------------------
ROWS = COLS = 4096
SCHEMES = [("int8_per_tensor", 8, None), ("int8_per_channel", 8, -1),
           ("int8_group128", 8, 128), ("int4_per_channel", 4, -1),
           ("int4_group128", 4, 128), ("int4_group32", 4, 32)]

for target_ratio in (5, 16, 64, 256):
    torch.manual_seed(1337)
    x = torch.randn(ROWS, COLS)
    if target_ratio > 6:                      # inject a sparse outlier population
        n_out = max(1, ROWS * COLS // 10_000)  # 0.01% of entries
        idx = torch.randint(0, ROWS * COLS, (n_out,))
        x.view(-1)[idx] = target_ratio * torch.sign(torch.randn(n_out))
    sigma = float(x.std()); ratio = float(x.abs().max()) / sigma
    print(f"\n=== target r={target_ratio}  measured outlier_ratio={ratio:.2f}  "
          f"sigma={sigma:.4f} ===")
    print(f"  {'scheme':22} {'rel_err':>10} {'predicted':>10} {'eff_bits':>9}")
    for label, bits, g in SCHEMES:
        deq, sb = q_int(x, bits, g)
        payload = ROWS * COLS * bits / 8
        # closed form from 3.2: RMS rounding error = delta/sqrt(12), delta = m_G/qmax
        m_g = (float(x.abs().max()) if g is None else
               float(x.abs().amax(-1).mean()) if g == -1 else
               float(x.reshape(ROWS, COLS // g, g).abs().amax(-1).mean()))
        pred = m_g / ((2 ** (bits - 1) - 1) * (12 ** 0.5) * sigma)
        print(f"  {label:22} {relerr(x, deq):10.6f} {pred:10.6f} "
              f"{(payload + sb) * 8 / (ROWS * COLS):9.4f}")
    for label, dt in [("fp8_e4m3fn_per_ch", torch.float8_e4m3fn),
                      ("fp8_e5m2_per_ch", torch.float8_e5m2)]:
        deq, sb = q_fp8(x, dt)
        payload = ROWS * COLS
        print(f"  {label:22} {relerr(x, deq):10.6f} {'—':>10} "
              f"{(payload + sb) * 8 / (ROWS * COLS):9.4f}")
```

**Predictions, written before you run.**

1. **The closed form tracks the measurement to within ~20%** for the integer schemes.
   `rel_err ≈ m_G / (qmax · √12 · σ)` — the mean group maximum, divided by the largest
   storable integer, divided by `√12` for the uniform-noise variance, normalized by σ.
   If it does not track, the noise is not uniform and the outliers are clipping.
2. **Error is flat in the outlier ratio for small groups and linear in it for large
   ones.** Per-tensor scaling sees the global maximum, so its error scales with `r`
   directly; group-32 scaling sees the max of 32 draws, which barely moves when the
   outliers are 0.01% of entries. This is §3.2's inequality as a curve.
3. **`int4_group32` beats `int8_per_tensor` once the outlier ratio exceeds roughly
   `127/7 ≈ 18`** — the ratio of the two `qmax` values — at *half* the bits. That
   crossover is the exercise's headline: it says granularity and bit depth are
   substitutable at a computable exchange rate, and it is why every "8-bit is safer than
   4-bit" intuition needs the grouping attached before it means anything.
4. `fp8_e5m2` (more exponent, less mantissa) degrades **more slowly** with the outlier
   ratio than `fp8_e4m3fn` and is worse at `r = 5`. Float formats buy range with
   resolution; that is the whole choice.

**Deliverables — one crossover, one plot, one sentence.**

1. The outlier ratio at which `int4_group32` overtakes `int8_per_tensor`. Compare against
   the predicted ≈18 and say whether the closed form was optimistic or pessimistic.
2. `rel_err` versus `eff_bits`, one series per outlier ratio, all eight schemes plotted.
   The **Pareto frontier** of that scatter is the only part of the quantization literature
   you need to hold in your head.
3. One sentence on what changes when you run this on real weights instead. (Hint: real
   weight outliers are not uniformly scattered — they concentrate in *channels*, which is
   why per-channel scaling is a distinct scheme on the list and why AWQ works at all.
   Synthetic scattered outliers understate the value of per-channel scaling and overstate
   the value of grouping along the input axis.)

**What a falsification would mean.** If error does not fall with group size, your scale is
being set by something other than the group max — check that `amax` reduces over the axis
you think it does. If the measured error is far *below* the closed form, your outliers are
being clipped rather than represented (check for saturation at `±qmax`), which is a
different quantizer than the one you think you wrote.

---

### Exercise B — FP8 is two number systems, and one of them is a trap on AMD

**Goal:** confirm §5.4 on our own instrument, in a fresh process, and re-confirm the
`_scaled_mm` unavailability that this whole module's framing rests on. This is the
cheapest genuinely load-bearing check in the module.

**Hardware:** CPU for the format probe, GPU for the `_scaled_mm` probe. **CPU fallback:**
the format probe is CPU-native; if your build lacks fp8 conversions on CPU, run it on GPU
and record *that* as the finding. **Runtime:** under a minute (estimate).

```python
"""FP8 format facts on this torch build. Deterministic; no timing."""
import torch

for n in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    print(n, torch.empty(0, dtype=getattr(torch, n)).element_size(), "byte/elem")

# All 256 bit patterns, read as two different number systems.
bits = torch.arange(0, 256, dtype=torch.uint8).view(torch.int8)
fn = bits.view(torch.float8_e4m3fn).to(torch.float64)
uz = bits.view(torch.float8_e4m3fnuz).to(torch.float64)
ok = torch.isfinite(fn) & torch.isfinite(uz) & (uz != 0)
print("distinct fn/fnuz value ratios:",
      sorted({round(float(r), 6) for r in fn[ok] / uz[ok]}))
print("max finite  e4m3fn:", float(fn[torch.isfinite(fn)].max()),
      " e4m3fnuz:", float(uz[torch.isfinite(uz)].max()))
print("bit pattern 0x80 ->  e4m3fn:", float(fn[128]), " e4m3fnuz:", float(uz[128]))

if torch.cuda.is_available():
    a = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn).t()
    s = torch.tensor(1.0, device="cuda")
    try:
        torch._scaled_mm(a, b, scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
        print("_scaled_mm: OK  <-- INSTRUMENT CHANGED, re-run the Hardware Validation Gate")
    except Exception as e:
        print("_scaled_mm:", type(e).__name__, str(e).splitlines()[0])
```

**Predictions.**

1. All four fp8 dtypes report **1 byte/element**. (Already `[M]` from an earlier probe;
   this reproduces it in a fresh process, which is what upgrades it from anecdote.)
2. The set of distinct `fn/fnuz` value ratios is **exactly `{2.0}`** — one element, no
   spread. That is `w8a8_utils.py:128`'s `* 2.0` justified from first principles rather
   than trusted.
3. Bit pattern `0x80` is **`0.0` as e4m3fn and `nan` as e4m3fnuz**
   (`w8a8_utils.py:116-117`).
4. `_scaled_mm` raises `RuntimeError` naming compute capability 9.0/8.9 or ROCm MI300+.

**`[M]` Reference numbers — this exercise WAS run**, unlike A and C. Z13 / gfx1151 /
native Windows, torch `2.12.0a0+rocm7.13.0a20260313`, `HIPBLASLT_TENSILE_LIBPATH` set,
`TORCH_BLAS_PREFER_HIPBLASLT=1`, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` unset,
fresh process, 2026-07-26. The probe enumerates all 256 bit patterns, so it is exact
rather than sampled and there is no seed to report:

```
float8_e4m3fn 1  float8_e5m2 1  float8_e4m3fnuz 1  float8_e5m2fnuz 1   byte/elem
distinct fn/fnuz value ratios: [2.0]                (252 of 256 patterns finite in both)
max finite  e4m3fn: 448.0    e4m3fnuz: 240.0        <-- NOT a 2:1 ratio; 1.867
bit pattern 0x80 ->  e4m3fn: -0.0    e4m3fnuz: nan
_scaled_mm: RuntimeError torch._scaled_mm is only supported on CUDA devices with
            compute capability >= 9.0 or 8.9, or ROCm MI300+
```

All four predictions held, and the run produced one thing none of them anticipated: **the
maxima differ by 1.867, not 2** (§3.3 explains why — FNUZ spends no encodings on ±inf and
only one on NaN). Predictions 1–4 were carried in this repo as an uncommitted scratch
anecdote before today; this run in a fresh process is what makes them `[M]`.

**Deliverables — two facts and a decision.**

1. The ratio set. If it is `{2.0}`, write it into your notes as the reason a
   NVIDIA-produced fp8 checkpoint cannot be byte-loaded on an MI300 and vice versa, and
   the reason gfx1151 wants the **OCP** variants (`rocm.py:890` — we are not gfx94x).
2. The exact `_scaled_mm` error string, with the wheel version. This is the fact the
   module's third headline rests on; it should be re-checked after every wheel change,
   because an upgrade is a change of instrument.
3. **Decide and record:** which fp8 dtype is Mnemosyne's default for a KV storage
   experiment on this machine, and why. (`e4m3fn`, per `rocm.py:890` plus prediction 2 —
   but write the argument down, because the alternative is a silent factor of two.)

**What a falsification would mean.** If the ratio set has more than one element, the two
formats differ by more than an exponent bias on this build and vLLM's single `* 2.0`
correction would be wrong here — a genuinely interesting finding and a notebook entry.

---

### Exercise C — the dequantize tax on weights, and why it is worst at batch 1

**Goal:** measure §3.8's prediction that unfused int4 weights are **~2.25× slower** than
bf16 at batch 1 and that the penalty **shrinks as batch grows**. This is the local test of
"we get memory economics and not tensor-core economics," on the weight side; the KV side
is already answered in `kv-cache-mechanics.md` Exercise C at 2.9–3.1× `[M]`.

**Hardware:** one gfx1151 GPU. **CPU fallback:** set `K = N = 2048` and `iters = 5`;
absolute throughput is meaningless but the *shape* of `slowdown(M)` survives, because it
is a traffic argument and the CPU reads the same DRAM. **Runtime:** ~4 minutes on GPU
(measured, six isolated subprocesses).

**Run each batch size in its own subprocess.** Not fastidiousness: one preparatory run of
this exact code died with a hipBLASLt access violation (§8, item 2) that a ten-cell
controlled retest could not reproduce. Isolating the cells costs a few seconds of process
startup and means a fault loses one number instead of the run.

**Footprint check first:** the largest tensor is the bf16 weight matrix at
`8192 × 8192 × 2 B = 134.2 MB`, plus a same-sized dequantized temporary. Both far inside
the `[M]` ≥62 GiB fast tier and far below the 31 GiB per-tensor hazard.

```python
"""Unfused int4 weights vs bf16 weights, at three batch sizes."""
import json, statistics, time, torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
K = N = 8192
GROUP = 128

W = torch.randn(K, N, dtype=torch.bfloat16, device=DEV)
q = torch.randint(0, 16, (K, N // 2), dtype=torch.uint8, device=DEV)      # 2 nibbles/byte
scales = torch.rand(K, N // GROUP, dtype=torch.bfloat16, device=DEV) + 0.5
lo_mask = torch.tensor(0x0F, dtype=torch.uint8, device=DEV)

def dequant():
    lo = (q & lo_mask).to(torch.bfloat16) - 8
    hi = (q >> 4).to(torch.bfloat16) - 8
    w = torch.stack((lo, hi), dim=-1).reshape(K, N)
    return (w.reshape(K, N // GROUP, GROUP) * scales.unsqueeze(-1)).reshape(K, N)

def sync():
    if DEV == "cuda": torch.cuda.synchronize()

def bench(fn, iters=20):
    for _ in range(3): fn()
    sync(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    sync(); return (time.perf_counter() - t0) / iters * 1e3

w_bytes = K * N * 2
q_bytes = K * (N // 2) + K * (N // GROUP) * 2
for M in (1, 8, 512):
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    t_bf16 = statistics.median(bench(lambda: x @ W) for _ in range(5))
    t_deq  = statistics.median(bench(dequant) for _ in range(5))
    t_int4 = statistics.median(bench(lambda: x @ dequant()) for _ in range(5))
    print(json.dumps(dict(
        M=M, bf16_ms=round(t_bf16, 4), dequant_only_ms=round(t_deq, 4),
        int4_path_ms=round(t_int4, 4), slowdown=round(t_int4 / t_bf16, 3),
        dequant_share=round(t_deq / t_int4, 3),
        storage_ratio=round(w_bytes / q_bytes, 3),
        bf16_implied_gbs=round(w_bytes / (t_bf16 / 1e3) / 1e9, 1))))
    del x
    if DEV == "cuda": torch.cuda.empty_cache()
```

**Predictions, as they were written before the run. Two held and two failed.**

1. `storage_ratio` ≈ **3.88×** — not 4×, because of the group-128 fp16 scales: the payload
   is 4 bits and the scale adds `16/128 = 0.125`, so the format is 4.125 bits/element and
   `16 / 4.125 = 3.879`. Deterministic; check it before you trust anything else in the
   output. (This module's first draft said 3.76 here, which is MXFP4's ratio at 4.25 bits.
   Wrong format, right idea, and exactly the kind of slip the printed number catches.)
2. `slowdown` at `M = 1` is **≈ 2.25** from the traffic argument, and probably somewhat
   worse because the unpack does real integer work per element (the fp8 KV analogue
   overshot its 2.5× prediction to 2.9–3.1× `[M]`).
3. `slowdown` **falls with `M`** and approaches 1 at `M = 512`, because dequantization is a
   fixed cost per weight matrix while useful FLOPs scale with `M`. **This is the exercise's
   headline and the sharpest falsifiable claim in the module.**
4. `bf16_implied_gbs` at `M = 1` lands in the 130–160 GB/s band, matching the
   decode-shaped attention reads in `kv-cache-mechanics.md` Exercise B rather than the
   199.9 GB/s copy figure — a second, independent confirmation that the copy benchmark is
   the wrong denominator for the ridge point.

**`[M]` Reference numbers — this exercise WAS run.** Z13 / gfx1151 / native Windows, torch
`2.12.0a0+rocm7.13.0a20260313`, `HIPBLASLT_TENSILE_LIBPATH` set,
`TORCH_BLAS_PREFER_HIPBLASLT=1`, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` unset,
`K = N = 8192`, group 128, seed 1337, median of 5 × 20 iterations per cell, **two
independent reps, each cell in its own fresh subprocess**, 2026-07-26:

| `M` | bf16 ms | dequant ms | int4 ms | **slowdown** | dequant share | bf16 GB/s |
|---|---|---|---|---|---|---|
| 1 | 0.6588 / 0.6578 | 8.077 / 8.095 | 8.644 / 8.847 | **13.12 / 13.45** | 0.934 / 0.915 | 203.7 / 204.1 |
| 8 | 0.6504 / 0.6605 | 8.050 / 8.067 | 8.853 / 8.842 | **13.61 / 13.39** | 0.909 / 0.912 | 206.3 / 203.2 |
| 512 | 3.7001 / 3.6593 | 8.105 / 8.139 | 10.978 / 11.128 | **2.97 / 3.04** | 0.738 / 0.731 | (compute-bound) |

`storage_ratio = 3.879` in all six cells, as predicted (1 ✓).

**Prediction 2 failed by 5.8×** and §3.8 explains why: the eager-mode dequantize
materializes four intermediates, not one, so it moves ~737 MB rather than ~168 MB. The
2.25× is the fused floor; 13× is the expression.

**Prediction 3 held, and the mechanism is confirmed rather than inferred:** `dequant only`
is flat at **8.05–8.14 ms across all three batch sizes** — a fixed per-matrix cost to
within 1% — while the bf16 arm rises from 0.66 to 3.70 ms, dragging the ratio from 13× to
3×. Note the sub-prediction that *did not* hold: it is not monotone between `M = 1` and
`M = 8` (13.1 vs 13.6, within run-to-run spread), because both are bandwidth-bound and the
bf16 side has not started to grow yet. The fall begins once the GEMM becomes compute-bound.

**Prediction 4 failed in the interesting direction:** 203–206 GB/s, *above* the 199.9 GB/s
copy figure and ~1.36× the ~150 GB/s of decode attention. See §3.8 — this localizes
attention's bandwidth deficit to its access pattern rather than to batch-1 matmul shape,
and it is a new number for `ASSUMPTIONS.md`.

Sanity check while you are here: at `M = 512` the bf16 arm does
`2 · 512 · 8192² = 68.7 GFLOP` in 3.70 ms = **18.6 TFLOPS**, against the `[M]` 20.9 TFLOPS
measured at 8192³. Consistent, and it tells you the `M = 512` cell really is compute-bound
so the `bf16 GB/s` column is meaningless there by construction.

**Deliverables — one curve and one decision.**

1. Plot `slowdown` against `M`. Report the `M` at which it crosses 1.0, or state that it
   does not within the range measured. (Ours does not; extrapolating the flat 8.1 ms
   dequantize against the compute-bound bf16 line puts the crossover near `M ≈ 1100`,
   which is arithmetic, not a measurement.)
2. `dequant_share` at `M = 1`. If it is above 50%, the cost is materializing temporaries,
   and the fix is a fused kernel, not a different format — the identical conclusion
   `kv-cache-mechanics.md` reached for fp8 KV at 65–66%. Ours is **93%**, which is that
   conclusion with the volume turned up.
3. **Write one line:** for a *capacity*-bound experiment on this machine, are int4 weights
   worth it? (Yes — **3.879× less resident** for a batch-1 speed penalty you are not
   measuring — and this is the same trade as fp8 KV. Say it explicitly, because the reflex
   from published throughput numbers is the opposite.)

**What a falsification would mean.** If `slowdown` at `M = 1` is near 1.0, torch is fusing
the dequantize into the GEMM prologue, which would be a genuinely new capability on this
stack and a change of instrument — record the wheel version and re-run the Hardware
Validation Gate. If it *rises* with `M`, the dequantize is not the fixed cost the model
says it is; check whether the temporary is being reallocated per call.

---

## 7. Self-check

Answers at the end of the file. Do not scroll.

1. A vendor claims their model is "4-bit quantized, 4× smaller." You have the GGUF and it
   is `q4_K`. Give the real compression ratio against bf16 and say where the missing
   compression went. Then say what would change if the format were MXFP4 instead.

2. Two teams quantize the same layer to int4. Team A uses per-output-channel scales; Team
   B uses group-32 scales along the input axis. Both report "int4." Which stores more
   bytes, and by how much, for a `4096 × 4096` matrix with fp16 scales? Which will have
   lower error, and what property of the weight matrix decides by how much?

3. You have a fixed KV byte budget. You can keep `N` tokens at fp8 or `2N` tokens at int4
   (ignore scales for a moment). Name the two arguments for each side and say why the
   question is currently open rather than settled.

4. A colleague benchmarks KV quantization using HuggingFace's `QuantizedCache` and reports
   that 4-bit KV is 6× slower than bf16 on our box. Name **three** distinct mechanisms in
   the measurement that are not the quantization, and rank them by how much of the 6× you
   expect each to explain.

5. An fp8 checkpoint produced on an MI300 is loaded byte-for-byte onto our gfx1151 box.
   The model runs and produces fluent text. What is numerically wrong, by how much, and
   which single line of vLLM would have prevented it?

6. Exercise C predicts the int4 weight penalty *shrinks* with batch size, while
   `kv-serving-hierarchy.md` shows KV read traffic does *not* amortize with batch size.
   Are these consistent? Use them to state, in one sentence, which of the two
   quantization decisions matters more in a high-batch long-context deployment.

---

## 8. What is still unsolved here

Everything below is testable at 20M–300M params on one GPU against a `[M]` ≥62 GiB fast
tier, unless marked otherwise. Each needs a pre-registered hypothesis card before it runs,
and all of them are blocked on the Hardware Validation Gate.

1. **The reference model weights are not on disk, and Exercise A is blocked on it.** All
   six model directories under `research/reference/models/` hold 135-byte Git LFS pointer
   stubs where the `*.safetensors` shards should be `[M]` (probed 2026-07-26; see
   Exercise A's preamble). Every `[M]` config read in this curriculum is unaffected, but
   **no exercise anywhere in the curriculum has ever touched a real trained weight**, and
   several planned ones need to. Fix `scripts/fetch_reference.sh` (LFS step, or an
   explicit exclusion so the absence is declared rather than discovered), then run
   Exercise A on Laguna-XS tensors and compare the real outlier ratios against the
   synthetic sweep. `[A]` High confidence the real ratios are higher and — more
   importantly — *channel-structured* rather than scattered, which would make per-channel
   scaling beat input-axis grouping in a way the synthetic version cannot show.

2. **An un-reproduced hipBLASLt access violation, reported here precisely because it did
   not reproduce.** While preparing Exercise C, one run died with
   `Exception Code: 0xC0000005` (exit 139) inside
   `libhipblaslt.dll → hipblasLtMatmulAlgoGetHeuristic`, reached from
   `at::cuda::mm → structured_addmm_out_cuda`, during the `M = 1`, `K = N = 8192`, bf16
   arm with `HIPBLASLT_TENSILE_LIBPATH` set. **A controlled retest immediately afterward
   ran the isolated `x @ W` in ten separate subprocesses — `M ∈ {1, 8, 512}` × hipBLASLt
   configured/unset at `K = N = 8192` bf16, plus `M = 1` at `K = N = 4096` bf16 and at
   fp32, both ways — and all ten exited 0** with agreeing outputs. So the bare operation
   is not the culprit, and **this is not `[M]`; it is one observation.** It is recorded
   because it is the *second* instance in this curriculum of a hipBLASLt crash that
   evaporated under isolation (see `curriculum/README.md` → the refuted skinny-K
   segfault), which starts to look like a pattern about the heuristic cache rather than
   about any one shape. Cheapest next test: run the full Exercise C script — not the bare
   GEMM — in `N` isolated subprocesses and count failures; an intermittent fault at, say,
   1-in-20 would be invisible to a single retest and lethal to a long run, exactly like
   the ≥32 GiB silent hang. Until then: **if Exercise C dies on you, that is a data point,
   not your mistake — record the rep index and the stack frame.**

3. **Nobody has measured the joint eviction × quantization budget at small scale.** `[C]`
   RDKV (2605.08317) and EvicPress (2512.14946) pose it; `[C]` 2607.08032 unifies it
   theoretically. At a fixed KV byte budget, is it better to keep more tokens at lower
   precision or fewer at higher? Two axes, matched budgets, one afternoon at our scale,
   and the answer is very likely task-dependent — which is itself the publishable finding
   if the crossover moves between a recall task and a reasoning task. **Highest
   information-per-GPU-hour item in this module.**

4. **The `residual_length` quality curve is unpublished.** §3.6: the staging buffer that
   makes per-channel key quantization streamable is also the amount of *recent* context
   held at full precision, and recent context is what the model attends to most. Sweep
   `r ∈ {0, 32, 128, 512}` at matched bytes. `[A]` High confidence the quality curve is
   steeper than the capacity curve near `r = 0`; cheapest test is exactly this sweep on a
   needle task.

5. **Is the KIVI key/value asymmetry real at 20M–300M params?** The claim is an empirical
   regularity of large trained decoders. Measure, on a model we trained ourselves,
   per-channel versus per-token outlier ratios for K and for V, per layer. A null result
   would mean small-scale KV-quantization ablations systematically mis-rank the methods —
   a confound affecting every arm this lab might run. It also composes with
   `kv-compression-and-eviction.md`'s open question 1 (the L2-norm anti-correlation),
   because both are asking whether a large-model regularity survives scale-down, and one
   forward pass answers both.

6. **Does quantization interact with our hybrid layer types?** Laguna is 12 global + 36
   sliding in strict GSSS `[M]`, with different RoPE θ and different partial-rotary
   fractions per layer type (`laguna.cpp:184`). `[C]` 2606.24033 argues bit allocation
   should be RoPE-aware because post-rotation keys have frequency-dependent sensitivity.
   If that is right, a hybrid stack needs *two* bit allocations, and nobody has looked.
   This is a Proteus-shaped question, not a Mnemosyne one, and it is ours to ask because
   our reference model has the property.

7. **Contested, and to be left contested: is FP4 deployable for anything but weights?**
   W4A16 is boring; W4A4KV4 is not. `[C]` 2603.08747 (Mar 2026) finds sensitivity strongly
   layer- and block-dependent; four separate recovery methods appeared in six months —
   `[C]` ARCQuant (2601.07475), `[C]` 2601.20088, `[C]` 2606.05682, `[C]` ReQAT
   (2606.15682). A field that needs four recovery methods that fast does not have a solved
   format. Present it as unsettled. **Not testable here** in the form the papers pose it
   (we cannot run FP4 arithmetic), though the *storage* half is.

8. **Contested: MXFP4 versus NVFP4.** The struct diff (§3.3) says the difference is finer
   groups plus a non-power-of-two scale for a quarter-bit. Whether that is worth it is
   `[C]` 2603.08747's subject and `[C]` 2605.12464's ("Search Your Block Floating Point
   Scales!") — i.e. the scale-selection rule is itself a research object. Vendor-reported
   NVFP4 KV numbers exist and are not independently replicated; the survey note flags them
   as claims, and so does this module.

9. **Is `[C]` 2605.05699's "quantization is free on bandwidth-limited hardware" claim true
   on gfx1151?** It is the closest published analogue to our platform — unified memory,
   bandwidth-limited, int4 KV outrunning fp16 — and our own fp8 KV measurement says the
   opposite by 3× `[M]`. Either their kernel fuses and ours does not (most likely), or the
   platforms differ in a way that matters. Resolving it needs a fused dequant kernel we do
   not have, which makes this the strongest argument in the repo for writing one Triton
   kernel by hand. **Cost that before attempting it.**

10. **The alignment-collapse effect at small scale.** `[C]` 2606.09864 tested 3.8B–72B
   instruction-tuned models. If the low-dimensional-subspace mechanism is real, a narrow
   behavioural probe should detect it at 300M, giving this lab a cheap attribution harness
   reusable by every later compression arm. Cross-listed from
   `research/memory/kv-cache-mechanics.md` open question 5 and
   `research/notes/inference-and-quantization.md` open question 6 — three documents now
   want this experiment, which is a reasonable signal that it should be scheduled.

11. **Nobody reports the scale overhead in a sub-4-bit compression claim.** §3.7 shows it
    is 12.5% at 2 bits. That is large enough to reorder published rankings. Re-deriving
    two or three headline sub-4-bit numbers with the scale tensor counted is a
    zero-GPU-hour methodological contribution and would take an afternoon with the papers
    open. Low glamour, high honesty-per-hour.

12. **`[M]` A new number that wants a home in `ASSUMPTIONS.md`, and a question behind it.**
    Exercise C measured **203–206 GB/s** for a batch-1 bf16 GEMV over a contiguous 134 MB
    weight matrix (six cells, two processes) — at or above the 199.9 GB/s device-to-device
    copy figure, and ~1.36× the ~150 GB/s that decode-shaped attention reads achieve `[M]`.
    Two consequences. First, the ridge point in `ASSUMPTIONS.md` (20.9 TFLOPS ÷ 199.9 GB/s
    ≈ 105) has a *different correct denominator per workload*: ~102 FLOP/byte for weight
    GEMV and ~139 for attention. Second, and more interesting: the attention deficit is
    now localized to attention's **access pattern** rather than to matmul shape, because a
    batch-1 GEMV is not slow here. `[A]` Medium-high confidence the mechanism is the
    strided gather across `n_kv` head slices; cheapest test is to time the same total bytes
    read as one contiguous tensor versus as `n_kv` strided slices, with no matmul at all.
    That is a twenty-line script and it closes open question 3 of
    `kv-cache-mechanics.md` §8.

---

## Answers to the self-check

**1.** `q4_K` is **4.5** bits/weight (`ggml-common.h:327`: 2 fp16 super-scales + 12 bytes
of 6-bit sub-scales + 128 bytes of nibbles = 144 bytes per 256 weights), so against bf16
the real ratio is `16/4.5 = ` **3.56×**, not 4×. The missing 11% went into the two-level
scale hierarchy, which exists so that the *effective* group is 32 rather than 256 — you
bought accuracy with it, and the vendor's "4×" quietly charged you for it twice by not
mentioning either. MXFP4 (`:215`) is 17 bytes per 32 values = **4.25** bits, ratio
**3.76×** — better compression, one power-of-two scale per 32 elements instead of a
hierarchy, and therefore worse error on outlier-heavy tensors. The comparison you want is
never "4-bit vs 4-bit"; it is bits/element against error, plotted.

**2.** Team A: one fp16 scale per output row = 4096 scales = 8 KiB. Team B: one per 32
inputs per row = `4096 × 128` = 524,288 scales = 1 MiB. Both store `4096²/2` = 8 MiB of
payload, so A is 8.008 MiB (4.008 bits/weight) and B is 9 MiB (**4.5** bits/weight) —
**B stores 12.4% more.** B will have lower error, and the margin is decided by the
**outlier ratio within a row**: from §3.2 the error floor is set by `m_G`, so if a row's
maximum is a lone spike far above its neighbours, A's single row-scale inflates `Δ` for
all 4096 entries while B's grouping confines the damage to the 32 entries sharing the
spike's group. If the row is homogeneous, B pays 12.4% more bytes for nothing. Exercise A
measures which regime Laguna's weights are in.

**3.** *For fp8 at `N` tokens:* per-element error is far lower, and fp8's dynamic range
handles the per-channel key outliers `[C]` KIVI documents without needing rotation; it is
also the production-boring choice, so kernel support is real. *For int4 at `2N` tokens:*
you keep twice the context, and `kv-compression-and-eviction.md` §1's exact error identity
says the dominant term is the *dropped attention mass* `(1 − A)` — evicted tokens
contribute error with weight `a_i`, while retained-but-noisy tokens contribute error
scaled by the quantization noise, and those are different quantities that nobody has
measured against each other. It is open because the two error sources live in different
units and no published harness reports both; `[C]` RDKV (2605.08317) is the first attempt
to put them under one objective, and it is recent enough to be unreplicated. Also
note that "ignore scales" is doing work in the question: with per-token-head fp32 scales
the int4 arm's real budget is 51 KiB/token against fp8's 96, so it is `1.88N`, not `2N`.

**4.** Three mechanisms, ranked by expected contribution:
   (a) **`torch.cat` maintenance amplification** (`cache_utils.py:724` plus
   `kv-cache-mechanics.md` §3.4). The quantized layer concatenates *three* tensors per
   step, on top of `DynamicLayer`'s already-`O(T²)` growth. At `T = 8192` that path alone
   was measured at **23× wall clock** against preallocation `[M]`. Expect most of the 6×
   here, and note it is not even a KV-quantization effect — the bf16 arm pays a
   `torch.cat` too, so what differs is the *third* concat and the dequantized temporaries.
   (b) **Full-cache dequantization every step** (`cache_utils.py:722`). §3.8's
   read-modify-write over the entire cache, per step, unfused — the direct analogue of the
   fp8 result at 2.9–3.1× `[M]` with dequant taking 65–66% of the path.
   (c) **No low-precision arithmetic on this hardware** — `_scaled_mm` unavailable `[M]`,
   so there was never a fast path to reach. This is a ceiling, not a cost, so it explains
   why the number cannot be *below* 1, not why it is 6.
   The honest answer to the colleague: that measurement is of the harness, and the
   experiment must be re-run with the same cache class on both arms — the rule
   `kv-cache-mechanics.md` §4.3 states as a one-line harness assertion.

**5.** MI300-class hardware is `gfx94x` and uses the **FNUZ** fp8 variants
(`rocm.py:890`, `interface.py:1081`); gfx1151 is not gfx94x and uses **OCP**
`e4m3fn`/`e5m2`. For an identical bit pattern the fnuz value is **half** the fn value, so
reading an fnuz checkpoint as fn makes every quantized tensor **2× too large** unless the
scales are halved — and separately, bit pattern `0x80` is a legitimate **zero** in fn and
a **NaN** in fnuz, so the conversion also has to remap that byte
(`w8a8_utils.py:116-121`). The single line that would have prevented it is
`w8a8_utils.py:128`, `weight_scale = weight_scale * 2.0`, applied in the direction
appropriate to the load. The reason the model still produces fluent text is the reason
this whole module keeps repeating: **there is no verify-on-read**, and a uniform 2× error
on weights is partially absorbed by the normalization layers, so the failure is a quality
regression with no error message.

**6.** They are consistent, and together they answer the question. Weight bytes are read
once per forward pass and shared by the whole batch, so both the weight *read* and the
unfused *dequantize* amortize as `1/M` — which is exactly what Exercise C measured `[M]`:
the dequantize is flat at 8.05–8.14 ms from `M = 1` to `M = 512` while the slowdown falls
from 13.1× to 3.0×. KV bytes are per-sequence and amortize over nothing, so at batch `B`
you read `B` caches. Therefore: **in a high-batch long-context deployment, KV quantization
dominates and weight quantization is close to irrelevant to throughput** (though still
decisive for whether the model fits at all). The corollary that catches people: a
quantization result measured at batch 1 and a quantization result measured at batch 32 are
answers to two different questions, and neither transfers. Note the second-order point
Exercise C also showed: the fall is not monotone at the low end — `M = 1` and `M = 8` are
indistinguishable because both are bandwidth-bound — so "amortizes with batch" begins only
once the GEMM crosses into compute-bound.

---

## Sources

**Local measurements and artifact reads (`[M]`).** Z13, Ryzen AI Max+ 395, Radeon 8060S
(gfx1151), native Windows, torch `2.12.0a0+rocm7.13.0a20260313` (HIP 7.2.0), venv
`C:\venvs\lab`, `HIPBLASLT_TENSILE_LIBPATH` set and `TORCH_BLAS_PREFER_HIPBLASLT=1` per
`scripts/activate-lab.ps1`, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` **unset**, all
2026-07-26. Scripts were run from a scratch directory and are **not committed**, so by
house standard these are instrument-shakedown numbers and not evidence until the Hardware
Validation Gate closes; §6 gives the code verbatim.

**New runtime measurements contributed by this module** (each reproduced in a fresh
process; the FP8 probe is an exhaustive enumeration with no seed, Exercise C used seed
1337):

- **FP8 format probe** (Exercise B, one fresh process): all four fp8 dtypes 1 byte/element;
  the set of `e4m3fn / e4m3fnuz` value ratios over the 252 bit patterns finite and nonzero
  in both is **exactly `{2.0}`**; max finite 448.0 (fn) vs 240.0 (fnuz), ratio **1.867**,
  not 2; bit pattern `0x80` is `-0.0` as fn and `NaN` as fnuz.
- **`torch._scaled_mm` unsupported** (same process): `RuntimeError: torch._scaled_mm is
  only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+`.
  This was previously carried across three repo documents on an uncommitted scratch probe;
  it is now reproduced.
- **Unfused int4 weight dequantize tax** (Exercise C, `K = N = 8192`, group 128, bf16,
  median of 5 × 20 iterations, **two reps × three batch sizes, each cell in its own
  subprocess**): slowdown **13.12 / 13.45** at `M = 1`, **13.61 / 13.39** at `M = 8`,
  **2.97 / 3.04** at `M = 512`; dequantize flat at **8.05–8.14 ms** at every `M`;
  `storage_ratio` 3.879 in all six cells.
- **Batch-1 bf16 weight-read bandwidth: 203.2–206.3 GB/s** (same six cells) — above the
  199.9 GB/s copy figure and ~1.36× the ~150 GB/s of decode attention. Candidate new
  `ASSUMPTIONS.md` row; §8 item 12.
- **Model weights absent** (probe, one fresh process): all six directories under
  `research/reference/models/` have 135-byte first shards beginning `"version http"` —
  Git LFS pointer stubs. §6 Exercise A preamble; §8 item 1.
- **Not tagged `[M]`, deliberately:** one hipBLASLt access violation (`0xC0000005`, exit
  139) inside `hipblasLtMatmulAlgoGetHeuristic` during an early combined Exercise C run.
  **A ten-cell controlled retest in isolated subprocesses — `M ∈ {1, 8, 512}` × hipBLASLt
  on/off at `K = N = 8192` bf16, plus `M = 1` at 4096 bf16 and 8192 fp32 both ways — exited
  0 in every cell.** One observation, not a measurement. §8 item 2. Recorded rather than
  deleted because a previous module in this curriculum tagged a non-reproducing crash as
  `[M]` and the correction is cheaper than the habit.

Everything else tagged `[M]` here is either (a) a read of a source file or config at the
revision pinned in `research/reference/PROVENANCE.md`, reproducible by opening the file,
or (b) a row carried from `ASSUMPTIONS.md` and attributed to the run that produced it.

- Artifact reads made for this module, all 2026-07-26:
  `ggml-common.h` block-size ledger (§3.3), the nine allowed KV cache types
  (`arg.cpp:301`), vLLM's `KVQuantMode` enum and scale accounting
  (`kv_cache_interface.py:33`, `:44`, `:180`, `:185`, `:208`), the four attention scale
  parameters and the per-tensor restriction (`kv_cache.py:65`, `:128`), the
  fn↔fnuz conversion (`w8a8_utils.py:110–131`) and its platform gate (`rocm.py:890`),
  the CUDA gating of Marlin (`auto_awq.py:312`, `fp8.py:287`), and the
  `QuantizedLayer` docstring's stated deviation from KIVI (`cache_utils.py:672`).
- Laguna-S parameter decomposition (§4.1): arithmetic over
  `research/reference/models/laguna-s/config.json` at revision `b0a9fd7c850e`
  (`ASSUMPTIONS.md → reference-model`). Experts ≈ 96.6% of parameters.
- Carried `[M]` rows from `ASSUMPTIONS.md`: `gpu-fast-tier-size` (≥62 GiB at ~200 GB/s,
  single run per arm), `large-tensor-fault-32gib` (≥32 GiB single tensors hang at 0% CPU
  or fault), `gemm-throughput-below-reference` (20.9 TFLOPS bf16 at 8192³),
  `hipblaslt-config` (2.8× change in long-reduction bf16 error; a numerics control),
  `bf16-reduced-precision-knob-works` (**refuted** — inert),
  `sdpa-is-memory-efficient` (**refuted by default** — 147.2 vs 6.6 bytes/T²),
  `bf16-numerics-unproven` (untested), `single-device-only`, `kv-per-token-laguna`
  (192.0 KiB/token exactly), `laguna-heads-uniform`, `torch-build`.
- Carried `[M]` from `curriculum/kv-cache-mechanics.md` Exercise C: fp8 KV storage with
  bf16 compute is **2.92–3.13× slower** than bf16 across six measurements in two
  processes and three context lengths, with dequantization taking 65–66% of the path.
- Carried `[M]` from `curriculum/kv-cache-mechanics.md` Exercise B: decode-shaped
  attention reads reach ~150 GB/s, not the 199.9 GB/s copy figure the ridge point uses.
- `torch._scaled_mm` unavailability on gfx1151 was previously recorded in
  `research/memory/kv-cache-mechanics.md` and `research/notes/inference-and-quantization.md`
  from a scratch probe that was **not committed** — an anecdote by house standard.
  Exercise B reproduced it in a fresh process for this module, so the three documents that
  depend on it now rest on a reproduced observation rather than a single one.

**Code pointers.** Every `file:line` in §5 was opened and the named construct confirmed on
the named line on 2026-07-26, against the revisions in
`research/reference/PROVENANCE.md`. Reused from `research/reference/CODE_MAP.md`
(machine-verified by `scripts/generate_code_map.py`):
`architecture/llama-cpp-laguna/src/llama-kv-cache.cpp:319`,
`memory/flashinfer/flashinfer/decode.py:1982`. Introduced by this module and verified by
reading:
`architecture/llama-cpp-laguna/ggml/src/ggml-common.h:90`, `:181`, `:188`, `:195`, `:202`,
`:215`, `:223`, `:252`, `:256`, `:276`, `:298`, `:327`;
`architecture/llama-cpp-laguna/common/arg.cpp:301`, `:2190`;
`architecture/llama-cpp-laguna/src/llama-kv-cache.cpp:313`, `:344`;
`architecture/transformers/src/transformers/cache_utils.py:672`, `:690`, `:722`, `:726`;
`memory/vllm/vllm/v1/kv_cache_interface.py:33`, `:44`, `:180`, `:185`, `:208`, `:209`;
`memory/vllm/vllm/model_executor/layers/quantization/kv_cache.py:42`, `:65`, `:85`,
`:128`, `:130`, `:150`;
`memory/vllm/vllm/model_executor/layers/quantization/utils/w8a8_utils.py:110`, `:116`,
`:128`;
`memory/vllm/vllm/platforms/rocm.py:888`, `:890`;
`memory/vllm/vllm/platforms/interface.py:1081`;
`memory/vllm/vllm/model_executor/layers/quantization/auto_awq.py:77`, `:193`, `:312`;
`memory/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:110`, `:118`, `:373`,
`:516`;
`memory/vllm/vllm/model_executor/layers/quantization/fp8.py:273`, `:287`;
`memory/flashinfer/flashinfer/decode.py:1989`.

**arXiv (`[C]`).** Every id below appears in the verified source lists of
`research/notes/inference-and-quantization.md` (66 ids queried against the live arXiv API
on 2026-07-26, 0 unresolved) or `research/memory/kv-cache-mechanics.md` /
`research/memory/kv-compression-and-eviction.md`. Resolving an id proves the paper exists,
not that it supports the claim beside it.

*Weight quantization — the lineage*
- `2208.07339` — *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* (2022-08-15). Outlier isolation.
- `2211.10438` — *SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs* (2022-11-18). The `xW = (xS⁻¹)(SW)` identity.
- `2210.17323` — *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (2022-10-31). The Hessian-weighted objective.
- `2306.00978` — *AWQ: Activation-aware Weight Quantization* (2023-06-01). Protect the ~1% of channels that matter, by scaling.
- `2404.00456` — *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs* (2024-03-30).
- `2405.16406` — *SpinQuant: LLM quantization with learned rotations* (2024-05-26).
- `2209.05433` — *FP8 Formats for Deep Learning* (2022-09-12). e4m3 / e5m2.
- `2310.10537` — *Microscaling Data Formats for Deep Learning* (2023-10-16). The MX/MXFP4 spec lineage.
- `2310.19102` — *Atom: Low-bit Quantization for Efficient and Accurate LLM Serving* (2023-10-29).

*FP4, and the reason it is contested*
- `2603.08747` — *Diagnosing FP4 inference: a layer-wise and block-wise sensitivity analysis of NVFP4 and MXFP4* (2026-03-05).
- `2601.07475` — *ARCQuant: Boosting NVFP4 Quantization with Augmented Residual Channels* (2026-01-12).
- `2601.20088` — *Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery* (2026-01-27).
- `2606.05682` — *Beyond Output Matching: Preserving Internal Geometry in NVFP4 LLM Distillation* (2026-06-04).
- `2606.15682` — *ReQAT: Full-Precision Reasoning Accuracy with 4-bit FP QAT* (2026-06-14).
- `2605.12464` — *Search Your Block Floating Point Scales!* (2026-05-12).
- `2606.06527` — *Characterizing the Impact of NVFP4 Quantization for Low-Power Edge AI Deployment* (2026-06-03).
- `2509.25149` — *Pretraining Large Language Models with NVFP4* (2025-09-29).
- `2603.10444` — *The Curse and Blessing of Mean Bias in FP4-Quantized LLM Training* (2026-03-11).
- `2607.04302` — *HiFA4: Training-Free 4-bit FlashAttention on Ascend HIF4 NPUs* (2026-07-05).

*KV-cache quantization*
- `2402.02750` — *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache* (2024-02-05). Keys per-channel, values per-token.
- `2401.18079` — *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization* (2024-01-31). Pre-RoPE keys, per-vector outlier isolation.
- `2510.05373` — *KVLinC: KV Cache Quantization with Hadamard Rotation and Linear Correction* (2025-10-06).
- `2605.05699` — *When Quantization Is Free: An int4 KV Cache That Outruns fp16 on Apple Silicon* (2026-05-07). The closest published analogue to our platform; our own fp8 result disagrees by 3×.
- `2606.09864` — *Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation* (2026-06-01). 15.2% of refusals lost at 1.03× perplexity.
- `2511.18643` — *Kitty* (2025-11). Dynamic channel-wise precision boost.
- `2606.03458` — *KVarN* (2026-06). Variance-normalised quantization for reasoning error accumulation.
- `2606.24033` — RoPE-aware bit allocation (2026-06). Post-rotation keys have frequency-dependent sensitivity.

*Joint budgets, and the framing this module adopts*
- `2605.08317` — *RDKV: Rate-Distortion Bit Allocation for Joint Eviction and Quantization of the KV Cache* (2026-05).
- `2512.14946` — *EvicPress: Joint KV-Cache Compression and Eviction for Efficient LLM Serving* (2025-12).
- `2607.08032` — *What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction in LLMs and Agents* (2026-07).
- `2607.04244` — *Quantize the Target, Quantize the Drafter* (2026-07-05). The quantization × speculation interaction as its own object.
- `2603.20397` — *KV Cache Optimization Strategies for Scalable and Efficient LLM Inference* (2026-03). No method dominates.
- `1911.02150` — *Fast Transformer Decoding: One Write-Head is All You Need* (2019-11-06). Why decode is bandwidth-bound.
- `2309.06180` — *Efficient Memory Management for LLM Serving with PagedAttention* (2023-09-12).

**Non-arXiv, and weaker for it**
- vLLM engineering blog, *The State of FP8 KV-Cache and Attention Quantization in vLLM*,
  22 Apr 2026 — https://vllm-project.github.io/2026/04/22/fp8-kvcache.html. The
  91% → 13% → 89% needle-in-a-haystack sequence, e4m3 with per-head scales, ~7k-token
  break-even, 13–15% throughput.
- ONNX float8 technical documentation (cited from `w8a8_utils.py:118`) — the normative
  statement of the fn/fnuz difference.
- ROCm documentation and LLVM commit traffic on scaled-WMMA target support (checked
  2026-07-26): the per-block-scaled FP4/FP6/FP8 matrix path lands on gfx1250 / RDNA4-class
  targets, **not** gfx1151.
- NVIDIA developer communications on NVFP4 KV cache (early 2026) — vendor-reported, single
  hardware family, not independently replicated. Deliberately not quoted as a result.

**Mirrored notes.** `research/notes/inference-and-quantization.md` v1.0.0 (§2–3) and
`research/memory/kv-cache-mechanics.md` v1.0.0 (§5) are the surveys this module teaches.
No number here contradicts either. This module adds six things they do not carry:

1. The per-format effective-bit ledger read from `ggml-common.h`, including `block_nvfp4`
   at 4.5 bits and `block_q1_0` at 1.125, neither of which appears in either survey (§3.3).
2. `[M]` The FP8 two-number-systems probe, with the finding neither the vLLM source
   comment nor the surveys state: per-bit-pattern values differ by exactly 2.0 while the
   *maxima* differ by 1.867 (§3.3).
3. The per-token scale-overhead table for Laguna-S — 3.13% at int8, 6.25% at int4, 12.5%
   at 2 bits (§3.7).
4. The Laguna-S capacity ledger showing that a quarter of a bit per weight decides whether
   the reference model's expert weights fit the measured ≥62 GiB fast tier (§4.1).
5. `[M]` The unfused int4 weight dequantize tax — 13.1–13.6× at batch 1, 3.0× at batch
   512, with the dequantize flat at 8.1 ms at every batch size (§3.8, Exercise C). This
   answers the weight-side analogue of `kv-cache-mechanics.md` open question 2, which had
   only been answered for KV.
6. `[M]` Batch-1 bf16 weight-read bandwidth at 203–206 GB/s versus decode attention's
   ~150 GB/s, which partially closes `kv-cache-mechanics.md` §8 open question 3 by showing
   the attention deficit is an access-pattern effect and not a small-`M` matmul effect
   (§3.8, §8 item 12).

It also promotes one framing to a design constraint: **Mnemosyne's write-time admission
hook must be able to return a precision, not just a keep/drop decision**, because eviction
and quantization spend the same budget (§4.2).

Two corrections were folded back into this file after its own exercises ran, rather than
being quietly applied: the batch-1 int4 slowdown prediction (2.25× → measured 13.1–13.6×,
with the eager-mode four-intermediate mechanism identified) and the batch-1 bandwidth
prediction (130–160 → measured 203–206 GB/s). Both are marked as failed predictions in
§6 and both changed a conclusion.
