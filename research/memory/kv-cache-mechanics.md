---
title: KV cache mechanics — shape math, bandwidth economics, and the reduction knobs
version: 1.0.0
date: 2026-07-26
track: research/memory (note 2 of 10)
---

# KV cache mechanics

This note settles three things. **One:** the per-token KV cost of a transformer is a
closed-form function of five config fields and nothing else, and once you write it out
you can size any long-context experiment on paper before you allocate a byte. **Two:**
the KV cache dominates long-context memory not as an empirical tendency but as an
identity — during decode it is the *only* term in the memory budget with a context
length in it — and it makes decode memory-bandwidth-bound in a way that no amount of
batching fixes. **Three:** every KV-reduction technique in the literature is a change to
exactly one factor in that formula, and knowing which factor tells you immediately what
it buys, what it costs, and whether two techniques compose.

Written for a reader who has capacity-planned storage tiers. The bridges are real, and
the places they break are flagged, because the breaks are where the ML content actually
lives.

---

## 1. The shape math

For one sequence, one token, the bytes the model must retain:

```
per_token_bytes = 2 × L × n_kv × d_h × b
```

| Symbol | Meaning | Where it comes from |
|---|---|---|
| `2` | one **K** tensor and one **V** tensor per token per layer | the attention definition; not a tunable |
| `L` | number of attention layers | `num_hidden_layers` |
| `n_kv` | key/value heads per layer (**not** query heads) | `num_key_value_heads` |
| `d_h` | dimension of one head | `head_dim` |
| `b` | bytes per stored element | `torch_dtype` (bf16 → 2, fp8 → 1) |

Note what is *absent*: the number of **query** heads, the hidden size, the vocabulary,
the MLP width, the expert count. None of them appear. A 118B MoE and a 300M dense model
with the same `L`, `n_kv`, `d_h` have identical KV cost per token. This surprises people
and it is the single most useful fact in the formula.

`[M]` **Worked on the reference model**, read from `research/reference/models/laguna-s/config.json`
(fetched at `b0a9fd7c850e`, see `PROVENANCE.md`): `L=48`, `n_kv=8`, `d_h=128`, `b=2` (bf16).

```
per-layer, per-token = 2 × 8 × 128 × 2 B = 4096 B = 4 KiB
all-48-layers        = 48 × 4 KiB        = 192 KiB / token
```

At 128k context that is **24.0 GiB**; at the model's advertised 1M
(`max_position_embeddings = 1048576`), **192 GiB**. Against the `[M]` **≥62 GiB** fast
memory tier measured on our Z13 (`notebook/uma-carveout-controls-fast-tier.md`), the
first fits with room and the second does not fit at all.

### Correcting a caveat already in our own register

`ASSUMPTIONS.md → kv-per-token-laguna` hedges this number because
`laguna-heads-uniform` found that Laguna varies head count per layer. `[M]` That hedge is
unnecessary, and the reason is worth internalising. In
`architecture/transformers/src/transformers/models/laguna/modeling_laguna.py`, line 346
takes `num_heads` per layer from `config.num_attention_heads_per_layer` (48 on global
layers, 72 on sliding ones) — but lines 357 and 360 build `k_proj` and `v_proj` with
width `config.num_key_value_heads * head_dim`, reading the **global** field. Query heads
vary; KV heads do not. Since query heads do not appear in the formula, the 192 KiB
figure is exact, not an upper bound.

Per-layer query head count is not irrelevant, though — it sets the GQA group size
`G = n_q / n_kv`, which is 48/8 = **6** on global layers and 72/8 = **9** on sliding
ones. Section 3 shows that `G` *is* the arithmetic intensity of decode attention. So the
per-layer head count controls speed, and the global `n_kv` controls size. Two different
questions, two different config fields, routinely conflated.

### Windowed layers split the formula into a growing term and a fixed term

Laguna-S is a 3:1 hybrid: `layer_types` gives 12 `full_attention` and 36
`sliding_attention` layers in a strict GSSS pattern, `sliding_window = 512` `[M]`. A
windowed layer never retains more than `w` tokens, so:

```
total(T) = (L_global × 2 × n_kv × d_h × b) × T   +   (L_window × 2 × n_kv × d_h × b × w)
           \___________ grows with T ___________/     \_______ constant in T _________/
```

`[M]` For Laguna-S: growing term **48 KiB/token**, fixed term **72 MiB** (36 × 4 KiB × 512).

| Context | All-global upper bound | Actual (hybrid) | Ratio |
|---|---|---|---|
| 32k | 6.00 GiB | 1.57 GiB | 3.8× |
| 128k | 24.00 GiB | 6.07 GiB | 4.0× |
| 1M | 192.00 GiB | **48.07 GiB** | 4.0× |

The hybrid ratio buys asymptotically `L/L_global` = 4×, and 1M-token Laguna-S KV lands
at 48 GiB — inside our 62 GiB tier, with 14 GiB left for everything else. That is a real
experiment we could run if we had the weights resident, which we do not; it is the
capacity-planning arithmetic that decides such things before anyone reboots.

**Where the storage analogy already breaks.** In llama.cpp's Laguna implementation the
two layer types get two *physically separate* caches, sized independently
(`architecture/llama-cpp-laguna/src/llama-kv-cache-iswa.cpp:73`, `size_swa`). That reads
like a two-tier hierarchy and it is not one: there is no promotion, no demotion, and no
miss path. A layer is bound to its tier forever by its index, and discarding an
out-of-window token is **lossless**, because `is_masked_swa` makes it architecturally
unreadable — the model could not have attended to it anyway. Tier-1 eviction here costs
nothing, which is true of no storage tier you have ever run.

---

## 2. Why the KV cache dominates long-context memory

The usual phrasing — "at long context the KV cache gets bigger than the weights" — is
true but weak, because it invites the reply "so buy more memory." The strong statement:

> **During decode, the KV cache is the only term in the memory budget that contains `T`.**

Weights: `P × b_w`, constant. Activations: one token's residual stream, constant. There
is no optimizer state. Everything except the KV cache is `O(1)` in context length, so
asymptotically the KV cache is 100% of the marginal memory. Whether it dominates *today*
is a question about where you sit on the curve, not whether the curve bends.

And the two terms have different sharing semantics, which is the part a serving engineer
should latch onto immediately: **weights are shared across every concurrent request; the
KV cache is private per sequence.** So the crossover is a function of batch as well as
context:

```
B* = (P × b_w) / (per_token_bytes × T + fixed_window_bytes)
```

`[M]` **Worked on gpt-oss-20b** (local artifact, `models/gpt-oss-20b/`): `L=24`
(12 full + 12 sliding at `w=128`), `n_kv=8`, `d_h=64`, so 2 KiB per layer per token →
**24 KiB/token** growing plus **3 MiB** fixed. Weights are 13,761,264,768 B = 12.82 GiB
(`model.safetensors.index.json → total_size`; the MoE is mxfp4 per `quantization_config`).
At its full 131,072-token context one sequence holds 3.00 GiB of KV — 23% of the weights.
`B* ≈ 4.3`, so **at batch 5 the KV cache outweighs the entire model.**

`[A]` **Worked at our own scale**, with the Proteus config assumed rather than frozen
(24 layers, `n_kv=8`, `d_h=64`, bf16, all-global attention, 300M params — medium
confidence; the cheapest thing that would move it is freezing an actual arm config):
weights 572 MiB, KV 48 KiB/token. At an 8k context, `B* ≈ 1.5` — **batch 2 and the cache
is bigger than the model.** At 32k × batch 8 the cache is 12 GiB against 0.56 GiB of
weights, a 21× ratio, and it fits our fast tier with room to sweep.

This is the finding that justifies the lab's hardware thesis in one line: at 20M–300M
params the weights are so small that we enter the KV-dominated regime at trivially short
contexts. We are not simulating the interesting regime; we are structurally inside it.
`[C]` The same asymmetry is what the 2026 serving literature is organised around — KV
and weights have different lifetimes and different sharing, and several 2026 systems
disaggregate them for exactly that reason (2607.02574, Jun 2026; 2607.08057, Jul 2026).

**Where the analogy breaks: this is a cache with no backing store.** Every cache you have
operated has a miss path — slower, but correct. The KV cache has none. Inside the
attention kernel a miss is *unrepresentable*: FlashInfer's page table has no present bit,
a page is in `kv_indices` or the token does not exist
(`memory/flashinfer/flashinfer/decode.py:1239`). vLLM's allocator, when it cannot find a
block, does not fault — it **preempts the whole request**
(`memory/vllm/vllm/v1/core/block_pool.py:647`). The only "reload" is re-running prefill
over the entire prefix. So eviction is destruction, the unit of eviction is the sequence
rather than the page, and the thing you would call hit rate is not a performance dial —
it is an accuracy dial.

---

## 3. Why decode is bandwidth-bound, in arithmetic

Inference has two phases with opposite characters `[C]` (2311.18677, Nov 2023;
2401.09670, Jan 2024): **prefill** processes the whole prompt at once — big matmuls,
compute-bound — and **decode** produces one token at a time. The relevant quantity is
**arithmetic intensity**: FLOPs performed per byte moved from memory. Compare it to the
machine's own ratio (peak FLOP/s ÷ peak byte/s, the roofline *ridge point*). Below the
ridge you are bandwidth-bound and the FLOP units idle; above it you are compute-bound.

`[M]` **Our machine's ridge point**, from measurements already in `ASSUMPTIONS.md`:
20.9 TFLOP/s bf16 GEMM at 8192³ (`scripts/benchmark_gemm.py`) ÷ 199.9 GB/s device-to-device
copy bandwidth at a 62 GiB footprint (`scripts/measure_memory_bandwidth_tiers.py`) =
**≈105 FLOP per byte**. Caveats stated honestly: the bandwidth figure counts one read plus
one write, a pure-read stream may differ; both are single-run; and the GEMM number is
itself only 63% of the ~33 TFLOPS cited for this silicon, unexplained
(`gemm-throughput-below-reference`). The ridge is therefore ~105 give or take, not a
precision instrument — but the gaps below are so large that precision does not matter.

### The weight-read term: arithmetic intensity equals batch size

Per decode step the model reads every weight once, no matter the batch, and performs
`2 × P` FLOPs per sequence:

```
AI_weights = (2 × P × B) / (P × b_w) = 2B / b_w    →  for bf16, AI = B
```

Batch size *is* the arithmetic intensity of the weight-bound half of decode. To reach our
~105 ridge you would need batch ≈105. `[A]` For a sparse MoE this is worse than it looks:
FLOPs scale with *active* params, but once the batch is wide enough that most experts get
selected by somebody, the *read* approaches the full parameter set. Laguna-S routes 10 of
256 experts `[M]`, so the read/compute ratio degrades by up to ~25× on expert layers as
batch grows. Medium confidence — it depends entirely on routing entropy, and the cheapest
test is instrumenting expert-selection cardinality per batch on a small MoE arm.

### The attention term: arithmetic intensity equals the GQA group size

This is the one that matters, and it is exact. For one layer, one sequence, context `T`:

```
bytes read  = 2 × n_kv × d_h × b × T                (the whole cache for that layer)
FLOPs       = 2·n_q·T·d_h   (QKᵀ)  +  2·n_q·T·d_h   (AV)  =  4 × n_q × T × d_h
```

Divide, and `T`, `d_h`, and `L` all cancel:

```
AI_attention = (4 · n_q · T · d_h) / (2 · n_kv · d_h · b · T) = 2·n_q / (n_kv · b) = 2G / b
```

**For bf16 (`b = 2`), the arithmetic intensity of decode attention is exactly the GQA
group size `G = n_q / n_kv`.** Independent of context length, head dimension, layer
count, and model size.

`[M]` For Laguna-S: `G = 6` on global layers, `9` on sliding ones. Against a ~105 ridge,
attention decode runs at **5.7%** and **8.6%** of peak compute respectively. Plain MHA
(`G = 1`) would sit at **0.96%** — consistent with the ~1–2 FLOP/byte figure widely quoted
for batch-1 decode `[C]`, and with the argument the MQA paper made in 2019 (1911.02150,
Nov 2019), which remains the cleanest single statement of why decode is a memory problem.

The mechanism is worth saying plainly: `G` query heads share one KV head, so one KV read
is amortised over `G` dot products. GQA raises arithmetic intensity by amortising along
the **head** axis exactly the way batching amortises the weight read along the **request**
axis.

**And this is why batching does not save you.** Each sequence owns its own KV cache, so
batching multiplies bytes and FLOPs by the same `B` and `AI_attention` does not move at
all. Weight-bound decode gets better with batch; attention-bound decode does not. As
context grows the attention term takes over, which is precisely why long-context,
high-batch serving is a bandwidth problem that more GPUs do not fix `[C]` (2607.13068,
Jul 2026, which formalises exactly this ridge-point mismatch and argues for decode
accelerators with less compute and more cheap memory).

**Where the analogy breaks — three ways.**

1. **Every access is a full scan.** A cache whose every read touches 100% of its contents
   has no temporal locality to exploit. Prefetching cannot help either: token `t+1`
   depends on token `t`, so the dependency chain is strictly serial. This is a streaming
   buffer wearing a cache's name.
2. **Read amplification is frozen at pretraining time.** `G` is an architecture decision
   baked into the checkpoint. There is no runtime knob, no re-sharding, no index rebuild.
3. **The one thing that *does* behave like a cache is on the other axis.** Prefix reuse —
   two requests sharing a system prompt — is a genuine hit-rate problem with a genuine
   working set, and it is governed by request routing rather than cache size. That is the
   subject of `kv-serving-hierarchy.md`, not this note.

---

## 4. GQA, MQA, MLA — the actual ratios

Read them as edits to the formula.

**MHA → GQA → MQA edits `n_kv`.** One knob, three names: `n_kv = n_q` is MHA,
`1 < n_kv < n_q` is GQA, `n_kv = 1` is MQA `[C]` (2305.13245, May 2023; 1911.02150,
Nov 2019). The reduction factor versus MHA is exactly `n_q / n_kv`, and — from Section 3
— it is *simultaneously* the arithmetic-intensity gain. One number, two wins. That
coincidence is why GQA won: it is the rare change that improves capacity and bandwidth by
the same factor with no extra machinery.

`[M]` Laguna-S against an MHA counterfactual (same `d_h`, `n_kv` set to each layer's own
query head count): MHA would cost `2 × 3168 × 128 × 2 = 1.547 MiB/token` versus the actual
192 KiB/token — an **8.25×** reduction. Pure MQA would give 24 KiB/token, another 8×
below where the model actually sits. The chosen `n_kv = 8` is a deliberate midpoint, and
`[C]` the GQA paper's argument for that midpoint is quality: MQA degrades, GQA at a
handful of groups largely does not, and an MHA checkpoint can be *uptrained* into GQA at
roughly 5% of pretraining compute (2305.13245).

**MLA edits the term structure, not `n_kv`.** Multi-head Latent Attention caches a
low-rank latent per token instead of per-head K and V, plus one shared RoPE key:
`d_c + d_h^R` elements per token per layer `[C]` (2405.04434, May 2024). DeepSeek-V2's
abstract claims **93.3%** KV reduction versus DeepSeek 67B, 42.5% lower training cost, and
5.76× maximum generation throughput.

`[M]` Verified against a local artifact — `models/kimi-linear-model/config.json` has
`kv_lora_rank = 512`, `qk_rope_head_dim = 64`, and `modeling_kimi.py:362` builds
`kv_a_proj_with_mqa` with output width exactly `kv_lora_rank + qk_rope_head_dim` = **576
elements/token/layer** = 1.125 KiB in bf16. Kimi Linear is a 3:1 hybrid with 7 MLA layers
of 27 (`linear_attn_config.full_attn_layers = [4,8,12,16,20,24,27]`), so **7.875 KiB/token**
— 7.9 GiB at 1M context, against Laguna-S's 48 GiB. Six times cheaper for a comparable
hybrid ratio.

**The gotcha that a paper will not tell you and the code will.** `[M]` In that same file,
`modeling_kimi.py:401` runs `kv_b_proj` to expand the latent back to full per-head K and V
*before* `past_key_values.update(...)` at line 413. The HF reference implementation
therefore caches 32 heads × (128+64) for K and 32 × 128 for V = **20 KiB per layer per
token** — **17.8× more than the compressed form, and zero saving from MLA.** The advertised
win exists only in a serving engine that caches `compressed_kv` and folds `kv_b_proj` into
the query side (the "absorb" trick). If you benchmark MLA using a reference
implementation, you will measure the opposite of the claim. This generalises: **the KV
cache is a property of the inference path, not of the architecture.**

**A third, orthogonal axis exists as of 2026.** MLA compresses each token's entry;
DeepSeek-V4's Compressed Sparse Attention and Heavily Compressed Attention compress along
the *sequence* axis, folding groups of `m` tokens into one entry, reported at ~7% of
V3.2's KV size at 1M context `[C]` (2606.19348, Apr 2026 — vendor-reported, not
independently replicated). Feature-axis and sequence-axis compression multiply. A 2026 KV
budget experiment that varies only one is under-designed.

**Contested, and to be presented as such.**
- *MLA versus GQA below frontier scale.* MLA's wins are reported at 100B+ MoE scale. As
  far as the anchoring survey pass could establish, whether the low-rank latent holds up
  at 20M–300M is **untested in public**. A conversion recipe exists that avoids
  pretraining two models `[C]` (2502.14837, Feb 2025: partial-RoPE removal plus joint SVD
  of K and V, recoverable on 0.3–0.6% of the data).
- *Cross-layer versus within-layer sharing.* Cross-layer KV sharing keeps producing
  papers, and the recurring finding is that it still tends to underperform within-layer
  GQA at matched budget. Do not present it as a strict improvement.
- *Adoption.* GQA at 4–8× remains the common industry choice in 2026; MLA is the default
  where DeepSeek- or Kimi-derived infrastructure is. Both camps are shipping.

---

## 5. FP8 KV quantization

This is the `b` factor, and arithmetically it is the least interesting knob in the note:
bf16 → fp8 halves per-token bytes, halves cache-read bytes, and therefore **doubles**
`AI_attention` (recall `AI = 2G/b`, so `b=1` gives `AI = 2G`). Laguna-S's global layers go
from 6 to 12 FLOP/byte. Everything else in the formula is untouched, so FP8 composes
cleanly with GQA, MLA, windowing, and eviction — it is the one reduction technique with no
interaction term in the shape math.

It is the *numerics* that are interesting.

**Formats.** `e4m3` (4 exponent bits, 3 mantissa) has better precision and a narrow
dynamic range; `e5m2` has more range and less precision. Production KV caches use `e4m3`
`[C]` (vLLM engineering blog, 22 Apr 2026), which is why a per-tensor, per-head, or
calibrated **scale factor** in higher precision travels alongside the quantized tensor —
the format alone cannot span activation dynamic range.

`[M]` **On our own instrument, today.** A probe on gfx1151 (torch
`2.12.0a0+rocm7.13.0a20260313`, HIP 7.2.0, native Windows) separates two questions people
routinely merge:

| | Result |
|---|---|
| FP8 as a **storage** dtype | `float8_e4m3fn`, `e5m2`, `e4m3fnuz`, `e5m2fnuz` all allocate on device at **1 byte/element** and round-trip through bf16. Median relative error on N(0,1) data: **2.19%** (e4m3), **4.35%** (e5m2). |
| FP8 as a **compute** dtype | `torch._scaled_mm` → `RuntimeError: only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+`. |

So on the Z13 an FP8 KV cache is available and buys the full capacity and bandwidth win,
but attention must dequantise to bf16 in-kernel: we get the memory economics and not the
FP8 tensor-core economics. That is fine — we are studying the memory economics. Public
sources disagree on whether RDNA 3.5 has native FP8 at all; our probe is the answer for
our stack and supersedes the argument. Worth re-running after every wheel change; it is a
change of instrument.

**The accuracy story is not "8 bits is basically free."** `[C]` The vLLM April 2026
writeup reports a 128k needle-in-a-haystack task falling from **91% (bf16) to 13%** under
FP8 KV plus FP8 attention on Hopper — traced not to the storage format but to imprecise
FP32 accumulation in the tensor-core path; a two-level accumulation fix restored **89%**
while keeping the speed (ITL slope at 54% of bf16, ~7,000-token break-even, 13–15%
throughput gain). Two lessons for this lab, both about attribution rather than outcome:
the failure was in the *kernel*, not the number format, and it only appeared at long
context, where quantization error accumulates over more terms. **If a result looks too
good, suspect the harness** — and if it looks catastrophically bad, suspect the harness
too.

**And perplexity will not catch the failure that matters.** `[C]` "Alignment Collapse
Under KV Cache Quantization" (2606.09864, 1 Jun 2026): across eleven instruction-tuned
models (3.8B–72B) and 1,894 prompts, Mistral-7B loses **15.2% of its refusals at 1.03×
perplexity**; there is no universal safe bit-width; and the stated root cause is
geometric — safety features occupy a low-dimensional activation subspace reported as
10²–10³× more sensitive to quantization noise than the average direction perplexity
integrates over. The authors confirm the vulnerability in production vLLM serving with
FP8 KV. This is the lab's stated weakness-of-the-literature in one artifact: outcome metric
fine, mechanism broken.

**Where the compression analogy breaks.** Storage compression is lossless and the bytes
come back bit-exact; the tradeoff is CPU for space. KV quantization is lossy, and the loss
is **structured**, not white. `[C]` Keys carry large per-channel outliers and values do
not, so the empirical rule every later quantizer inherits is quantize keys per-channel and
values per-token (2402.02750, Feb 2024); pre-RoPE key quantization, sensitivity-weighted
datatypes, and per-vector outlier isolation are what get you below 4 bits (2401.18079,
Jan 2024). Rotating the data before quantization to smear outliers across blocks is now a
standard trick and appears in real inference code, not just papers — llama.cpp's Laguna
branch applies a Hadamard rotation to K and V before a quantized cache store
(`architecture/llama-cpp-laguna/src/llama-kv-cache.cpp:319`), gated by a runtime heuristic
rather than by anything in the model config. `[M]` Note that the same branch has **no FP8
KV path at all** — its quantized-KV story is ordinary block quantization, which is a useful
corrective to assuming FP8 is universal.

**Contested — the bit-depth frontier.** 8-bit/FP8 is production-boring and 4-bit is
broadly safe. Sub-4-bit is not settled: KIVI and KVQuant claim near-lossless 2-bit
(2402.02750; 2401.18079), while 2025–26 work reports substantial degradation on reasoning
and generation, with attention-sink destruction and error accumulation over long chains
offered as mechanisms — and new entrants exist precisely because the problem is open
(Kitty, 2511.18643, dynamic channel-wise precision boost; KVarN, 2606.03458,
variance-normalised quantization for reasoning error accumulation). The honest 2026
summary is task-dependent: perplexity-friendly, reasoning-hostile. Related and also live:
whether bit allocation should be RoPE-aware, since post-rotation keys have
frequency-dependent sensitivity (2606.24033).

---

## 6. What to carry forward

1. **Five config fields set the whole capacity story.** `2 × L × n_kv × d_h × b`. Query
   heads, hidden size, and expert count do not appear.
2. **`G = n_q / n_kv` is the decode arithmetic intensity in bf16.** Same number that sets
   the GQA capacity win. It is 6 and 9 on our reference model, against a ~105 ridge.
3. **Batching fixes the weight term and never the attention term.** Long context plus wide
   batch is a bandwidth problem by construction.
4. **The KV cache belongs to the inference path, not the architecture.** MLA in the HF
   reference implementation saves nothing. Measure the cache, do not read it off the
   config.
5. **FP8 halves `b` and doubles arithmetic intensity, and the risk is in numerics you are
   probably not measuring.** Perplexity is not a safety metric.

---

## Open questions

Testable at 20M–300M params, single GPU, ≥62 GiB fast tier, no working multi-GPU. These
feed the Ablation Backlog; each needs a pre-registered hypothesis card before it runs.

1. **Does the `AI = 2G/b` prediction hold on gfx1151?** Sweep `n_kv ∈ {1,2,4,8,n_q}` at
   fixed `n_q` and measure achieved decode bandwidth and tokens/s. The model predicts a
   linear speedup in `G` until the ~105 ridge, then flatness. A deviation localises to the
   attention kernel rather than the architecture. Cheapest experiment in this note and it
   validates the instrument at the same time.
2. **Does the fp8-storage / bf16-compute path actually recover the bandwidth win on this
   hardware?** Measured `_scaled_mm` is absent, so the dequantise happens somewhere. If
   the dequant is not fused into the attention kernel, FP8 could cost more than it saves.
   Directly measurable with a decode-shaped microbenchmark under 32 GiB per buffer.
3. **Does MLA beat GQA at matched parameter count and matched KV budget below 300M?**
   Public evidence is at 100B+ scale only. Matched-KV-budget rather than matched-rank is
   the fair framing, and the 2502.14837 conversion recipe makes the arm cheap.
4. **Where is the `G` cliff at our scale?** GQA quality claims come from 7B–70B uptraining.
   At 20M–300M the per-head capacity is far smaller, so the group size at which quality
   breaks may be much lower — which would mean small-scale ablations systematically
   *understate* how good GQA is, a confound affecting every arm we run.
5. **Does the FP8 alignment-collapse effect have a small-scale analogue?** 2606.09864
   tested 3.8B–72B instruction-tuned models. If the low-dimensional-subspace mechanism is
   real it should be detectable in a 300M model with a narrow behavioural probe, which
   would give this lab a cheap attribution harness for every later compression arm.
6. **What is the ridge point for the *attention kernel* specifically, not GEMM?** Our ~105
   FLOP/byte uses an 8192³ GEMM and a copy benchmark. A decode-shaped attention roofline
   would give the number that actually governs, and would resolve whether the 63%-of-cited
   GEMM shortfall also afflicts attention.
7. **Does sequence-axis compression compose multiplicatively with feature-axis
   compression, as the shape math implies?** Both are single factors in the same product,
   so the null hypothesis is clean multiplication and any interaction is a finding.

---

## Sources

**Verified against the arXiv API in `research/reference/papers/anchors.bib`** (dates are
the arXiv `published` field):

- `1911.02150` — *Fast Transformer Decoding: One Write-Head is All You Need* (2019-11-06). MQA; the origin of the decode-is-bandwidth-bound argument.
- `2104.09864` — *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021-04-20). Why cached keys are position-encoded and reordering them is not a metadata operation.
- `2305.13245` — *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023-05-22). The `n_kv` knob and the ~5%-of-pretraining uptraining recipe.
- `2309.06180` — *Efficient Memory Management for LLM Serving with PagedAttention* (2023-09-12).
- `2311.18677` — *Splitwise: Efficient generative LLM inference using phase splitting* (2023-11-30). Prefill compute-bound / decode bandwidth-bound.
- `2401.09670` — *DistServe: Disaggregating Prefill and Decoding* (2024-01-18).
- `2401.18079` — *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization* (2024-01-31).
- `2402.02750` — *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache* (2024-02-05). Keys per-channel, values per-token.
- `2403.19887` — *Jamba: A Hybrid Transformer-Mamba Language Model* (2024-03-28). KV-cache economics as the justification for hybrids.
- `2405.04434` — *DeepSeek-V2* (2024-05-07). MLA; the 93.3% / 42.5% / 5.76× claims.
- `2412.19442` — *A Survey on LLM Acceleration based on KV Cache Management* (2024-12-27).
- `2502.14837` — *Towards Economical Inference: Enabling MLA in Any Transformer-based LLMs* (2025-02-20). The GQA→MLA conversion recipe.
- `2510.26692` — *Kimi Linear: An Expressive, Efficient Attention Architecture* (2025-10-30).
- `2606.19348` — *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence* (2026-04-26). Sequence-axis compression.
- `2607.02574` — *From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving* (2026-06-30).
- `2607.08057` — *Towards Efficient LLM Serving: A Survey on System-Aware KV Cache Optimization* (2026-07-09).

**Verified by fetching the arXiv abstract page on 2026-07-26:**

- `2312.04985` — *SparQ Attention: Bandwidth-Efficient LLM Inference* (2023-12-08, rev. 2024-09-04). Up to 8× reduction in attention data transfer; the clearest statement that bandwidth and capacity are separable targets.
- `2606.09864` — *Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation* (2026-06-01). 15.2% refusal loss at 1.03× perplexity.
- `2607.13068` — *The Economics of AI Decoding Chips: Rebalancing Compute, Capacity, and Bandwidth for Efficient LLM Inference* (2026-07-10). Formalises the ridge-point (F/B) mismatch for decode.
- `2603.02188` — *Multi-Head Low-Rank Attention* (2026-03-02). MLA's tensor-parallel sharding bottleneck; relevant only if we ever leave single-device.

**Cited from live search results on 2026-07-26; ids resolve on arxiv.org but were not
read in full.** Flagged as such rather than presented as read:

- `2511.18643` — *Kitty: Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost*.
- `2606.03458` — *KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks*.
- `2606.24033` — *RoPE-Aware Bit Allocation for KV-Cache Quantization*.

**Non-arXiv:**

- vLLM engineering blog, *The State of FP8 KV-Cache and Attention Quantization in vLLM*, 22 Apr 2026 — https://vllm-project.github.io/2026/04/22/fp8-kvcache.html (e4m3, per-head scales, the 91%→13%→89% needle-in-a-haystack sequence, 54% ITL slope, ~7k-token break-even, 13–15% throughput).

**Local artifacts and measurements** (the `[M]` claims above):

- `research/reference/models/laguna-s/config.json`, `laguna-xs/config.json`, `gpt-oss-20b/config.json`, `gpt-oss-20b/model.safetensors.index.json`, `kimi-linear-model/config.json` — fetched revisions recorded in `research/reference/PROVENANCE.md`.
- `research/reference/models/kimi-linear-model/modeling_kimi.py:362`, `:399`, `:401`, `:413` — MLA cache width and the decompress-before-cache path.
- `research/reference/architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:346`, `:357`, `:360` — per-layer query heads versus global KV heads.
- `research/reference/CODE_MAP.md` — verified pointers for llama.cpp Laguna two-tier KV (`llama-kv-cache-iswa.cpp:73`), Hadamard rotation before quantized KV store (`llama-kv-cache.cpp:319`), vLLM block allocation (`block_pool.py:647`), FlashInfer page-table plan API (`decode.py:1239`) and NHD/HND layout (`page.py:403`).
- `ASSUMPTIONS.md` rows `gpu-fast-tier-size` (≥62 GiB at ~200 GB/s), `gemm-throughput-below-reference` (20.9 TFLOPS bf16 at 8192³), `large-tensor-fault-32gib`, `reference-model`, `kv-per-token-laguna`, `torch-build`.
- FP8 dtype probe on gfx1151, 2026-07-26, torch `2.12.0a0+rocm7.13.0a20260313` — storage OK at 1 byte/element for all four FP8 variants; `torch._scaled_mm` unsupported ("ROCm MI300+"). Single run, scratch script; **not yet migrated into the rig or committed**, so it is an anecdote by the house standard and should be re-run as a Hardware Validation Gate item.
