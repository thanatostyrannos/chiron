---
title: Hybrid attention and ratios — a two-tier store whose tier assignment is a compile-time constant
version: 1.0.0
date: 2026-07-26
track: C — Memory (the deep track)
prereqs: attention-variants-and-kv-cost (hard), transformer-forward-pass-by-hand, tensors-and-autograd
mirrors: research/memory/hybrid-architectures.md
difficulty: 3/5 on the arithmetic, 4/5 on the evidence appraisal
time: 3–4 h reading and working the math; 2–3 h for the three exercises
---

# Hybrid attention and ratios

**Difficulty and time, honestly.** The arithmetic is one sum split into two terms and you
will have it in twenty minutes. What takes the afternoon is the evidence appraisal in §2.5
and §8: eight labs shipped eight different numbers, four of them ran real ablations, and
every ablation that reported a quality surface reported a *flat* one. Learning to hold
"this was ablated" and "the ablation showed the ratio barely mattered" in the same
sentence is the actual skill this module teaches. Budget 3–4 hours for §1–§5 with a pen,
and 2–3 hours for the exercises. Exercise B is the one to protect time for: writing it
produced a repeatable, *silent* process death on this machine, and the first explanation for
it was wrong in a way worth studying.

---

## 1. What this module settles

**One.** A hybrid decoder is a genuine two-tier store — the expensive tier is the
full-attention layers, whose residency grows as `O(T)`, and the cheap tier is the windowed
layers, whose residency is pinned at `O(w)` — but the asymptotic capacity win is the
*period* of the pattern, `L / L_global` = ratio **plus one**, and below a crossover at
`T = ratio × window` the window, not the ratio, is the term that dominates residency.

**Two.** The published ratios are not folklore and they are also not optimizations: four
teams ran real ablations, and every ablation that reported a quality surface reported a
*flat* one, so the shipped point was chosen by whatever memory-or-throughput tie-breaker
the team used — while the one metric on which the surface is *not* flat, recall, is the
metric nobody selected on.

**Three.** The tiering analogy earns its keep right up to the point where it inverts:
enlarging the fast tier (widening the window) makes the slow tier (the global layers)
*worse*, because the two tiers are co-trained rather than independently provisioned, and no
storage hierarchy you have ever operated behaves that way.

This module mirrors `research/memory/hybrid-architectures.md`. That note surveys; this one
teaches, and adds three things the note does not have: the crossover `T× = r·w` and its
consequences for our ablation scale (§3.4), the receptive-field arithmetic that explains
why *placement* is a separate axis from *ratio* (§3.7), and three further ways the tiering
analogy breaks that are only visible in the weight shapes and the serving code (§2.3).
Where I extend the note I say so; nothing here contradicts it.

---

## 2. Theory in plain language

### 2.1 Four knobs wearing one name

"3:1 hybrid" collapses four independent decisions. Separate them before you compare
anything, because the literature does not.

| Knob | What it is | Values in the wild |
|---|---|---|
| **Efficient primitive** | what replaces full attention in the cheap layers | sliding-window softmax attention (SWA); a constant-state recurrence (Mamba-2, Gated DeltaNet, KDA, xLSTM); block-sparse attention |
| **Fusion granularity** | where the two primitives meet | **inter-layer** (a layer is one type or the other); **intra-layer / head-wise** (both run in the same layer on the same input); **weight-tied** (one shared attention module invoked on a schedule) |
| **Ratio** | cheap units per expensive unit | 1:1 to 15:1; 3:1 is the current fashion |
| **Window / state size** | the cheap layer's own memory budget | SWA windows 128–4096 tokens; recurrent states a few hundred KB per layer |

Two traps, both of which have produced wrong cross-paper comparisons in print.

**The denominator moves.** Jamba's "1:7" is attention:Mamba. Kimi Linear's "3:1" is
linear:full. Gemma 3's "5:1" is local:global. Same colon, different fraction. Read the
direction before you read the number.

**The primitive changes what the ratio means.** A 3:1 SWA:global model still runs softmax
attention in every layer — the cheap layers just have a bounded receptive field, and their
KV cost is `O(w)`, not `O(0)`. A 3:1 linear:full model has *no attention at all* in three of
four layers and a genuinely constant-size state. These are different architectures and
there is no reason their optimal ratios should coincide. §3.6 prices both at the same shape
so you can see how different they are.

### 2.2 The systems bridge, stated properly

You have provisioned two-tier storage. Hot tier: fast, small, expensive per byte. Cold
tier: slow, large, cheap. You size them by working-set analysis, you set a placement
policy, you measure hit rate, and you accept that a miss costs a fetch.

A hybrid decoder really is that, and the correspondence is not a metaphor — it is the
literal allocation. llama.cpp's Laguna path constructs **two separate `llama_kv_cache`
objects**: a full-size one filtered to the non-SWA layers, and a small one whose capacity is
`min(size_base, n_swa·n_seq + n_ubatch)` padded to 256 cells
(`architecture/llama-cpp-laguna/src/llama-kv-cache-iswa.cpp:69` and `:73`) `[M]`. vLLM goes
further and runs several block tables over one physical pool, one per attention type
(`memory/vllm/vllm/v1/core/kv_cache_utils.py:1140`) `[M]`. The capacity arithmetic in §3 is
capacity planning in the ordinary sense, and your instincts about it are correct.

Then it breaks, in six specific places. Three are in the survey note; three more become
visible only once you read the weight shapes and the serving layer. The breaks are the
content of this module.

### 2.3 The six breaks

**Break 1 — no promotion, no demotion, no miss path.** A layer is bound to one tier forever
by its index, decided from a config key at load time
(`architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:365`) `[M]`.
Nothing migrates. There is no working-set analysis to do at runtime because there is no
runtime decision.

**Break 2 — "eviction" from the cheap tier is lossless, not a bet.** An out-of-window token
is architecturally *unreadable*: the mask forbids it. Discarding it costs exactly nothing.
Contrast an H2O or SnapKV eviction, which is a wager that a token will not be needed `[C]`
(2306.14048, 2023; 2404.14469, 2024). **A window is a proof; an eviction policy is a bet.**
That distinction is why windowing and eviction are not the same subject and must never
share an ablation axis.

**Break 3 — the tiers are not numerically interchangeable.** Laguna's global layers apply
YaRN-scaled RoPE over 64 of 128 head dimensions at θ=500000; its SWA layers apply plain RoPE
over all 128 at θ=10000 (`architecture/llama-cpp-laguna/src/models/laguna.cpp:184`) `[M]`.
You therefore cannot "just widen the windows" to test long context — the SWA layers were
never trained with a positional encoding that reaches past 512. This kills the obvious
experiment, and it kills it silently: you will get numbers, and they will be measuring the
positional encoding failing, not the ratio.

**Break 4 — the tier assignment is not a property of placement, it is a property of the
object.** In storage, a tier is where a byte lives; move the byte and you have re-tiered it.
Here the two tiers have *different weight shapes*. Laguna's sliding layers carry 72 query
heads and its global layers 48 `[M]` (`config.json` at `b0a9fd7c850e`,
`num_attention_heads_per_layer` = 48 on the 12 `full_attention` layers, 72 on the 36
`sliding_attention` ones; read per layer at `modeling_laguna.py:343`). In the llama.cpp
branch the same asymmetry appears for XS.2 at 48 versus 64 heads `[M]`
(`research/reference/CODE_MAP.md`, llama-cpp-laguna section). Re-tiering a layer would mean
reshaping its parameters. There is no re-tiering operation, even in principle, without
retraining — which is also the deepest reason the post-hoc-search papers in §8 have to
*freeze weights and re-select layers* rather than move anything.

**Break 5 — hit rate is an intersection across tiers, not a per-tier property.** This one
is worth the read. vLLM's full-attention manager finds a prefix hit by scanning **left to
right from token 0** and breaking on the first miss, because the block hashes are chained
(`memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:708`) `[M]`. Its sliding-window
manager scans **right to left**, looking for a contiguous run of cached blocks long enough
to fill the window
(`memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:918`) `[M]`. Prefix-matching and
suffix-matching over the same request. The engine must serve *one* boundary, so
`HybridKVCacheCoordinator.find_longest_cache_hit` reconciles them with an explicit
**fixed-point iteration** — each group either accepts the current candidate length or
reduces it, and any reduction restarts the loop over all groups
(`memory/vllm/vllm/v1/core/kv_cache_coordinator.py:685`, loop at `:727`) `[M]`. It
terminates only because the candidate length is monotonically decreasing and bounded below
by zero. In a storage hierarchy, a hit in *any* tier is a hit. Here, **a hybrid's prefix
cache hit rate is bounded by its least-cacheable tier**, and the general case is not even
implemented: the docstring states plainly that `find_longest_cache_hit` "only supports one
attention type or two types of full-attention plus exactly one another type" and that the
authors "don't know how to implement it cleanly yet"
(`memory/vllm/vllm/v1/core/kv_cache_utils.py:1191`) `[M]`.

**Break 6 — the escape hatch only runs one way.** Both production engines ship a switch that
*abandons* the hybrid saving, and neither ships the inverse. llama.cpp has `swa_full`, which
sets `size_swa = size_base` and logs a warning
(`architecture/llama-cpp-laguna/src/llama-kv-cache-iswa.cpp:76`) `[M]`. vLLM has
`unify_hybrid_kv_cache_specs`, which promotes every `SlidingWindowSpec` to a
`FullAttentionSpec` and warns that "we do not enable any optimizations for saving KV cache
memory… The compute of layers like sliding window is still saved"
(`memory/vllm/vllm/v1/core/kv_cache_utils.py:1547`, warning at `:1562`) `[M]`.

Think about why the inverse cannot exist. Promotion to the expensive tier is always
*correct*, because a full cache is a superset of a windowed one and the attention mask
discards the extra. Demotion is not correct, because you cannot serve a global layer from a
windowed cache — the bytes are gone and there is nowhere to fetch them from. **The only
legal runtime move on this hierarchy is the one that costs money.** Every storage system you
have run has the opposite asymmetry: you can always spill down, and spilling up is the
optimization.

**And the deepest break, which motivates the whole Mnemosyne programme: a hybrid never
knows it missed.** There is no hit-rate counter, no fault, no error return, no signal of any
kind that the windowed layers dropped something the model needed. A cache that silently
returns approximate data on a miss is not a cache; it is an approximation with no
observability surface. Every measurement in this track has to be constructed, because the
architecture emits nothing.

### 2.4 What hybrids replaced, and the one claim that is well supported

Before hybrids you picked a corner. Pure full attention pays `O(T)` residency in every
layer. Pure SWA or pure recurrence pays `O(w)` or `O(1)` everywhere and loses exact
long-range lookup. The hybrid claim is that a small number of exact layers recovers most of
the capability at most of the saving.

**That claim is the best-supported thing in this track.** Jamba at 1.3B/250B tokens and
7B/50B tokens `[C]` (2403.19887, Mar 2024); MAD across 500+ models from 70M to 7B `[C]`
(2403.17844, Mar 2024); the 72-model open sweep at 340M/20B and 1.3B/100B `[C]` (2507.06457,
Jul 2025, rev. Jun 2026). Three independent groups, three scales, same direction: hybrid
beats either pure architecture at matched budget.

Note what that claim is *not*. It says a hybrid beats a pure model. It says nothing about
which ratio.

### 2.5 Where the ratios came from — the ledger

`research/memory/hybrid-architectures.md` §4 builds this ledger by classifying each source
on **what the source document itself says**, not on what a later summary claims for it.
Read that table in full. The compressed version, plus one row this module adds:

| Model / study | Ratio shipped | Evidence class |
|---|---|---|
| Jamba `[C]` 2403.19887 | 1 attn : 7 Mamba | **Ablated** — and 1:3 ≈ 1:7, so 1:7 was chosen for compute, not quality |
| Gemma 3 `[C]` 2503.19786 | 5 local : 1 global, w=1024 | **Ablated** — swept 1:1 to 7:1 and windows 1024/4096, reported *minimal* perplexity impact; chosen on memory |
| Kimi Linear `[C]` 2510.26692 | 3 KDA : 1 MLA, NoPE on MLA | **Ablated** — 0:1 / 1:1 / 3:1 / 7:1 / 15:1; 3:1 best. See the caveat in the note: the primary table was not fetchable |
| Rope-to-Nope `[C]` 2501.18795 | 1 global : 3 local, w=4096 | **Ablated** — 1:1, 1:3, 1:7 tested; "a 1:3 ratio strikes an optimal balance." NoPE on global layers, RoPE on local *(row added by this module)* |
| 72-model sweep `[C]` 2507.06457 | recommends 3:1–6:1 | **Ablated** — 6 linear variants × 5 ratios × 2 scales, all open-sourced. The only large controlled academic sweep |
| Hymba `[C]` 2411.13676 | 3 global of 32 layers, parallel heads | **Ablated** incrementally — Table 1 is a seven-step A→G roadmap with a measured delta per change |
| Ring-linear `[C]` 2510.19338 | 4:1 at 16B, 7:1 at 104B | **Fitted** — scaling-law curves per layer-group size. Same team, same recipe, *different ratio at different scale* |
| Nemotron-H `[C]` 2504.03624 | ~8% attention layers, evenly dispersed | **Inherited, and says so** — cites Waleffe et al. `[C]` 2406.07887 |
| gpt-oss `[C]` 2508.10925 | 1:1 alternating banded, w=128 | **Inherited, and says so** — attributes the pattern to GPT-3 precedent |
| MiMo-V2-Flash `[C]` 2601.02780 | 5:1 SWA:global, w=128 | **Undisclosed** |
| Qwen3-Next / Qwen3.5 `[C]` (model cards) | 3 Gated DeltaNet : 1 full | **Undisclosed** |
| **Laguna S 2.1** `[M]` (our fetch) | 3 SWA : 1 global, w=512 | **Undisclosed** — there is no paper; the ratio is readable only from the shipped artifact |

Four things follow, and the third is the one to remember.

1. **Real ablations exist.** "Hybrid ratios are folklore" is false as stated. What is true is
   narrower and more interesting.
2. **Every ablation reporting a quality surface reports a flat one.** Gemma 3: minimal
   perplexity impact from 1:1 through 7:1. Jamba: 1:3 ≈ 1:7. Kimi Linear: on the order of
   0.01–0.05 PPL across a 15× span of ratios. 2507.06457: language modelling stable across
   ratios. When the objective is flat, the selected point is determined by the tie-breaker,
   which is usually memory or throughput. That is a perfectly good way to choose, but it is
   not the claim "3:1 is optimal."
3. **Recall is where the surface is not flat.** 2507.06457's core finding: language
   modelling is stable across the linear:full ratio, but **recall degrades sharply once
   full-attention layers thin below 3:1** `[C]` (rev. Jun 2026). So perplexity-selected
   ratios were selected on the one metric structurally unable to see the failure.

   *A correction to our own register, made while writing this module.*
   `research/memory/memory-failure-register.md → hybrid-ratio-sensitivity` lists 2510.26912
   as independent corroboration of the recall cliff. Reading its abstract (2026-07-26), that
   is not what the paper is about: *Understanding and Enhancing Mamba-Transformer Hybrids for
   Memory Recall and Language Modeling* (2025-10-30) compares **sequential versus parallel
   integration** of SSM and attention and proposes paraphrase-augmented training; it studies
   recall, but along the fusion-granularity axis, not the ratio axis `[C]`. It is a real and
   relevant result — see §2.6(c) — but it does not corroborate the ratio cliff, and
   2507.06457 should be treated as the single source for that claim until something else
   replicates it.
4. **Ratio may be scale-dependent.** Ring-linear ships 4:1 at 16B and 7:1 at 104B from the
   same fitting procedure `[C]` (2510.19338). If that generalises, every fixed-ratio claim
   is scale-local and a 300M rig must report its scale loudly.

### 2.6 Three findings more robust than the ratio

**(a) Longer windows *hurt* long context.** Two independent 2026 results, different groups,
different primitives. SWAX (SWA + xLSTM) finds larger windows degrade long-context
performance, because short windows *force* the model to train its long-term memory instead
of leaning on local softmax; too-small windows hurt short context, so they train with a
**stochastic** window and beat fixed windows on both `[C]` (2509.24552, Sep 2025, rev. May
2026). "Large-Window Laziness" finds larger SWA windows *delay the formation of retrieval
heads* in the full-attention layers `[C]` (2606.15378, Jun 2026).

This is the counterintuitive one and it is where the tiering metaphor dies:
**increasing the fast tier's capacity degrades the slow tier's behaviour**, because the two
tiers are not independently provisioned — they are co-trained, and the cheap one crowds out
the expensive one's learning signal. There is no storage-hierarchy analogue. There is not
even a bad one.

**(b) The ratio may be a training-speed knob, not a capability ceiling.** 2606.15378's
scaling analysis finds different hybrids converge to comparable long-context performance
given enough training, with the efficient-attention design controlling how *fast* the
capability emerges. Its constructive result: apply NoPE to only the full-attention layers of
a small-window SWA hybrid and long context improves substantially at negligible
short-context cost. Note that this is exactly what Kimi Linear ships (NoPE on all MLA
layers) `[C]` (2510.26692) and exactly what Rope-to-Nope ablated its way to a year earlier
`[C]` (2501.18795). Three independent arrivals at the same design is stronger evidence than
any one of them.

**(c) Placement matters, and uniform is probably not optimal.** 2510.04800's systematic
comparison of inter-layer versus intra-layer fusion reports, for a 1:12 Mamba hybrid, that
"placing the Transformer block in the early layers leads to significant performance drop,
while positioning it in the middle yields the best results", with the stated mechanism that
early Transformer blocks "exhibit a highly uniform attention distribution across all
tokens, functioning essentially as a global context encoder", which clashes with Mamba's
local inductive bias. It also recommends "aim for a high 1:1 ratio [for quality], but to
balance with efficiency, use about 1:5", and reports that intra-layer hybrids "take the
pareto-frontier of model quality and efficiency" `[C]` (2510.04800, Oct 2025, rev. Apr 2026;
read from the arXiv HTML render on 2026-07-26).

*This extends the survey note*, which flagged that it could not extract 2510.04800's
per-axis conclusions and said to read the paper before citing a winner. These are the
conclusions, read from the HTML render rather than the PDF; treat them as one careful
reading of one paper, not as settled.

**A second, independent paper points the same way on the fusion axis.** 2510.26912
(2025-10-30) compares sequential against parallel SSM/attention integration and reports that
"sequential hybrids perform better on shorter contexts, whereas parallel hybrids are more
effective for longer contexts" `[C]`. Same month, different group, and consistent with
2510.04800's Pareto claim for intra-layer fusion. Two papers is not a consensus, but it is
better evidence than the ratio question has, and note what it implies: **the axis the whole
ratio literature does not vary may be the axis that matters.** Every production model in
§2.5 is inter-layer.

Note also the tension with §3.7 below: Laguna
front-loads its first global layer at index 0 (`set_swa_pattern(4, dense_first=true)`,
`architecture/llama-cpp-laguna/src/models/laguna.cpp:41`) `[M]`, which is precisely what
2510.04800 says never to do — for a different primitive, at a different ratio.

---

## 3. The math that actually matters

### 3.1 Symbols, every one translated

| Symbol | Reads as | Where it comes from |
|---|---|---|
| `T` | tokens currently in context | runtime |
| `L` | total layers in the stack | `num_hidden_layers` |
| `L_g` | layers that are **global** (full attention) | count of `full_attention` in `layer_types` |
| `L_s` | layers that are **windowed** (sliding) | `L − L_g` |
| `r` | the advertised ratio, `L_s / L_g` — cheap layers per expensive layer | derived, never configured |
| `w` | sliding window, in tokens | `sliding_window` |
| `n_kv` | key/value heads per layer | `num_key_value_heads` |
| `d_h` | width of one head's key or value vector | `head_dim` |
| `b` | bytes per stored element (2 for bf16) | `torch_dtype` |
| `c` | **bytes per token per layer**, `c = 2 · n_kv · d_h · b` | derived; the constant everything else multiplies |

`c` is the whole of the previous module compressed into one letter. For Laguna:
`c = 2 · 8 · 128 · 2 B = 4096 B = 4 KiB` `[M]`. Nothing in this module changes `c`; every
result here is about what multiplies it.

### 3.2 The residency split

```
bytes(T)  =  c · [  L_g · T   +   L_s · min(T, w)  ]
                  └─ growing ─┘   └──── fixed ────┘
```

- The **growing term**, `c · L_g · T`, is the expensive tier. Linear in context, forever.
- The **fixed term**, `c · L_s · w`, is the cheap tier. It grows until `T = w` and is then
  constant for the rest of time.

Worked on Laguna `[M]` (`L=48`, `L_g=12`, `L_s=36`, `w=512`, `c=4 KiB`; config at revision
`b0a9fd7c850e`):

```
growing:  12 × 4 KiB           = 48 KiB per token
fixed:    36 × 4 KiB × 512     = 72 MiB, constant past 512 tokens
```

At 128k context that is 6.0 GiB + 0.07 GiB ≈ **6.07 GiB**, against **24.0 GiB** if every
layer were global. This is stated and derived in `attention-variants-and-kv-cost.md` §3.5
and in the survey note; it is repeated here only because everything below multiplies it.

### 3.3 The asymptotic saving is the period, not the ratio

```
                 L · T                        L
saving(T)  =  ───────────────────    ──────→  ───  =  r + 1     as T → ∞
              L_g·T + L_s·min(T,w)             L_g
```

For Laguna, 48/12 = **4×**, not 3×. **A "3:1 hybrid" saves 4×.** If you quote 3× you are
wrong by a third, and you will be wrong in the same direction every time, because the naming
convention counts the cheap layers and the saving counts all of them.

For `T ≥ w` the exact finite-context form is worth carrying in your head:

```
                 (r + 1) · T
saving(T)  =  ─────────────────
                  T  +  r·w
```

Sanity check the two ends. At `T = w`: saving `= (r+1)w / (w + rw) = 1`. Correct — a hybrid
saves **nothing** at or below the window. As `T → ∞` the `r·w` becomes negligible and the
expression → `r+1`. Correct.

### 3.4 The crossover: `T× = r · w`

Set the growing term equal to the fixed term and solve.

```
c · L_g · T   =   c · L_s · w
        T     =   (L_s / L_g) · w   =   r · w
```

**Below `T× = r·w`, most of the KV cache is in the cheap tier; above it, most is in the
expensive tier.** At exactly `T×` the split is 50/50. This is trivial arithmetic and I could
not find it stated in any of the hybrid papers, which is why it is worth stating: it is the
number that tells you *which knob you are actually tuning* at your target context.

| Model (at its shipped `r`, `w`) | `T× = r·w` | 90 % of asymptotic saving, at `9·r·w` |
|---|---|---|
| gpt-oss (1:1, w=128) `[C]` 2508.10925 | **128** | 1,152 |
| MiMo-V2-Flash (5:1, w=128) `[C]` 2601.02780 | **640** | 5,760 |
| Laguna S 2.1 (3:1, w=512) `[M]` | **1,536** | 13,824 |
| Gemma 3 (5:1, w=1024) `[C]` 2503.19786 | **5,120** | 46,080 |
| Rope-to-Nope (3:1, w=4096) `[C]` 2501.18795 | **12,288** | 110,592 |

The 90 % column comes from setting `saving(T) = 0.9(r+1)` in §3.3: `(r+1)T = 0.9(r+1)(T+rw)`
→ `T = 9·r·w`. Check it on Laguna: at T = 13,824, saving = `4 × 13824 / (13824 + 1536)` =
55296/15360 = 3.6, which is 90 % of 4. ✓

Three consequences.

**There is no context-free answer to "which schedule is cheaper."** Take two real shipped
configurations at the same depth `L = 48`: Laguna's (3:1, w=512) and gpt-oss's (1:1, w=128).
Setting their residencies equal:

```
T  =  (L_s^B·w_B − L_s^A·w_A) / (L_g^A − L_g^B)
   =  (24·128 − 36·512) / (12 − 24)
   =  (3072 − 18432) / (−12)
   =  1280 tokens
```

Below 1280 tokens gpt-oss's schedule holds **less** KV than Laguna's; above it, more. A
paper comparing them at 2k and a paper comparing them at 512 would reach opposite
conclusions about which is the memory-efficient design, and both would be right.

**Our ablation scale sits awkwardly close to `T×`.** At `w = 512` and a 2k training context —
a plausible Proteus arm — a 3:1 arm holds `12·2048 + 36·512 = 43,008` token-units and a 7:1
arm holds `6·2048 + 42·512 = 33,792`. That is a 1.27× byte difference across a ratio change
of 2.3×. **A ratio ablation scored on residency or on decode wall-clock is nearly blind at
our context lengths.** If the ablation is about memory, we need `T ≫ r·w`, which at `w=512`
and `r=7` means `T ≳ 32k`. If it is about recall — which is what §2.5 says it should be —
context length is set by the recall probe, not by the byte budget, and the memory argument
is simply not the reason to run it. Knowing which of those two experiments you are running,
before you run it, is the point.

**It also tells you which knob a paper was really tuning.** gpt-oss at `T× = 128` is a
model whose entire cheap tier is irrelevant to capacity past the first sentence; its 1:1
pattern is a compute decision. Rope-to-Nope at `T× = 12,288` is a model whose windows
dominate residency across most of its evaluation range; its ratio is nearly a compute
decision too. Only Laguna and MiMo sit where the ratio is genuinely a capacity decision at
their operating context.

### 3.5 What the ratio costs in bandwidth, on our instrument

Decode must stream the entire resident cache once per generated token. At the `[M]` 199.9
GB/s device-to-device figure measured on a 31 GiB buffer
(`ASSUMPTIONS.md → large-tensor-fault-32gib`, single run):

```
Laguna hybrid @128k :  6.07 GiB = 6.518 GB  ÷ 199.9 GB/s  ≈  32.6 ms/token  ≈ 31 tok/s
all-global @128k    : 24.00 GiB = 25.77 GB  ÷ 199.9 GB/s  ≈  129  ms/token  ≈  8 tok/s
```

That is arithmetic over two measured inputs, not a benchmark, and it ignores weight traffic
(for a 118B-A8.5B MoE, roughly 17 GB/token at bf16, which dominates until context passes
~100k). The systems point survives: **the ratio is a bandwidth decision before it is a
quality decision**, and at long context it is the dominant one.

Exercise B tests the prediction directly, and the short version of what it found on this
machine is: **the 3:1 arm's measured time ratio reproduces at 3.78–3.79 against a byte ratio
of 3.95, and the thinner arms do not reproduce at all.** Read §6 Exercise B before quoting
any of the wall-clock numbers above as though they were measured.

### 3.6 The two "3:1"s are not the same architecture — price them at matched state

The cheap tier of an SWA hybrid holds `c · w` bytes per layer. The cheap tier of a
linear/SSM hybrid holds a fixed-size state with **no `T` in it at all** `[C]` (2312.00752,
2023; 2405.21060, 2024) — for Mamba-2 the allocation is literally `(batch, nheads, headdim,
d_state)` with no sequence dimension
(`architecture/mamba/mamba_ssm/modules/mamba2.py:352`) `[M]`.

Put concrete numbers on both.

```
SWA layer, Laguna's shape:     c · w  =  4 KiB × 512                    =  2.0 MiB / layer
Mamba-2 layer, d_model=3072:   nheads · headdim · d_state · b_state
                               d_inner = 2 × 3072 = 6144;  headdim 64  → nheads = 96
                               96 × 64 × 128 × 4 B (fp32 state)        =  3.0 MiB / layer
```

`[A]` Medium confidence — the Mamba-2 defaults `d_state=128`, `expand=2`, `headdim=64` are
read from `architecture/mamba/mamba_ssm/modules/mamba2.py:41` and the two lines below it
`[M]`, and the fp32 state from `states_in_fp32=True` at
`architecture/mamba/mamba_ssm/ops/triton/ssd_combined.py:375` `[M]` — the chunk states are
computed in fp32 regardless of model dtype, which makes the "constant state" quietly the most
numerically fragile part of the layer, and matters directly for bf16 validation on unproven
hardware. `d_model = 3072` is my choice, to match Laguna's `hidden_size` `[M]`. Cheapest test
that would move this: instantiate both layers and read `untyped_storage().nbytes()`.

Two things to take from that.

**At these shapes they are the same order of magnitude.** 2 MiB versus 3 MiB per layer. The
"constant-state models use dramatically less memory" story is a story about *very long*
contexts, and even then the SWA layer's cost is also constant past `T = w`. The real
difference is not size; it is that the SWA layer's 2 MiB is an **exact** record of the last
512 tokens while the Mamba layer's 3 MiB is a **lossy** summary of everything, degrading by
interference rather than by eviction (`CODE_MAP.md`, Gated DeltaNet section).

**So the honest comparison axis is recall at matched state bytes, not ratio at matched
parameters.** That is the recall-versus-state-size Pareto frontier `[C]` (2402.18668, 2024),
and it is the frame the survey note recommends and that the ratio literature mostly does not
use. Two arms both labelled "3:1" can differ by 50 % in runtime state and by an unbounded
amount in what they can retrieve.

### 3.7 Receptive field: why placement is a separate axis from ratio

Here is the arithmetic that explains the placement literature, and it is arithmetic nobody
seems to report.

A windowed layer lets information move at most `w − 1` positions forward. Stack `k` windowed
layers and the output at position `t` can be influenced by source positions no earlier than
`t − k·(w−1)`. So:

```
reach(k windowed layers)  =  k · (w − 1)  +  1        source positions
```

A **single global layer sets reach to unbounded for itself and every layer above it**.
Therefore, at fixed `L_g`, placement controls two quantities that the ratio does not touch:

- **`below`** — how many layers sit under the first global layer. Those layers are hard-capped
  at `below · (w−1) + 1` positions of reach, no matter how long the context is.
- **`above`** — how many layers sit above the last global layer. Those layers can *process*
  long-range information but cannot *fetch* any more of it.

Work it for `L = 48`, `L_g = 12`, `w = 512`, so `w − 1 = 511`:

| Placement | Global layer indices | layers below first | reach of those layers | layers above last |
|---|---|---|---|---|
| `uniform-dense-first` (Laguna GSSS) | 0, 4, 8, … 44 | 0 | unbounded from layer 0 | 3 |
| `uniform-dense-last` (SSSG) | 3, 7, … 47 | 3 | 1,534 | 0 |
| `front-loaded` | 0–11 | 0 | unbounded from layer 0 | 36 |
| `back-loaded` | 36–47 | 36 | **18,397** | 0 |
| `middle-loaded` (2510.04800's recipe) | 18–29 | 18 | 9,199 | 18 |
| `hymba-style` (first/middle/last blocks) | 0–3, 22–25, 44–47 | 0 | unbounded from layer 0 | 0 |

Read the `back-loaded` row carefully. At a 32,768-token context, the bottom 36 layers can
reach at most 18,397 positions back — **the first 14,371 tokens of the context are
physically invisible to three quarters of the stack.** Not "attended to weakly". Invisible.
No training budget fixes that; it is a graph reachability property of the schedule.

And now the tension worth sitting with. Receptive-field arithmetic says front-load: get a
global layer in early so nothing downstream is reach-limited. 2510.04800 says the opposite
for Mamba hybrids at 1:12 — never put the attention block at the front, put it in the middle
`[C]`. Laguna front-loads (`dense_first=true`, `laguna.cpp:41`) `[M]`. Gemma 3, gpt-oss and
Nemotron-H disperse. Hymba uses first/middle/last. **Nobody agrees, the mechanisms proposed
are different mechanisms, and the receptive-field number above is not reported by any of
them.** Exercise C computes it, which is enough to make it an ablation axis in Proteus.

---

## 4. Why it matters for Proteus and Mnemosyne

**The config surface is the experimental surface, literally.** Laguna's hybrid schedule is a
list indexed by layer number, resolved once at construction
(`modeling_laguna.py:365`) `[M]`. Proteus must expose `layer_types` as an **explicit list**,
never a modulo. Samba shows why: it recomputes the schedule from `layer_idx % mb_per_layer`
inside the block constructor (`architecture/samba/lit_gpt/model.py:323`) and then makes a
*second, independent* modulo decision about windowed-versus-global inside the attention
module (`architecture/samba/lit_gpt/model.py:452`) `[M]`. Two sources of truth for one
property, in two classes, with no shared definition. That is exactly the bug shape that
makes an ablation unreproducible six weeks later. Samba's own escape hatch is an explicit
position list (`architecture/samba/lit_gpt/model.py:321`) `[M]` — start there.

**Four ablation axes, not one.** `layer_types` (which layers), `sliding_window` (`w`),
placement at fixed `L_g`, and the efficient primitive. §3.4 says `r` and `w` interact
through `T×`; §3.7 says placement is orthogonal to both. An arm named `proteus-swa-3to1` is
under-specified — the house naming rule requires the information to be in the name, so the
arm is `proteus-swa-3to1-w512-densefirst`.

**Match state bytes across arms, not parameters.** §3.6: a 3:1 SWA arm and a 3:1 linear arm
at equal parameters differ by 50 % in runtime state at our reference shape. Report both
numbers in every arm's manifest, computed from the config, and fail the run if they differ
by more than the pre-registered tolerance.

**Never report a hybrid-ratio result on perplexity alone.** §2.5, point 2. Score recall
explicitly — MQAR-style multi-query associative recall `[C]` (2312.04927, 2023) and a
RULER-style synthetic battery `[C]` (2404.06654, 2024). A flat perplexity surface is the
*expected* result and carries no information.

**The minimum context for a memory-motivated ratio ablation is `T ≫ r·w`.** From §3.4. At
`w = 512` and ratios up to 7:1 that is `T ≳ 32k`, which against the `[M]` ≥62 GiB fast tier
is affordable at 300M — but it is a deliberate choice, not a default, and it should be
written into the hypothesis card.

**Instrument for attribution, because the architecture emits nothing.** Three counters
Mnemosyne has to construct, none of which exists in any implementation read for this module:

1. **Per-layer attention mass beyond `w`** on the global layers — how much probability the
   expensive tier is placing where the cheap tier cannot see. This is the closest thing to a
   hit rate that a hybrid admits.
2. **Reach-limited fraction** — from §3.7, the fraction of the context that is unreachable
   by each layer under the current schedule. Purely a function of the config, so it costs
   nothing and belongs in the arm manifest.
3. **Retrieval-head formation over training steps** — "Large-Window Laziness" was findable
   only because someone measured *when* retrieval heads formed, not *whether* the model was
   good `[C]` (2606.15378).

**Hardware fit, and one hard constraint.** 1M-token Laguna KV is 48.07 GiB, inside the
`[M]` ≥62 GiB fast tier with room
(`notebook/uma-carveout-controls-fast-tier.md`, single run per arm). But single tensors
≥32 GiB hang silently at 0 % CPU `[M]` (`ASSUMPTIONS.md → large-tensor-fault-32gib`), so a
long-context KV cache must be allocated in sub-32 GiB chunks — which blocked/paged
allocation gives for free, and which is one more reason Mnemosyne should be paged from the
start rather than retrofitted.

**Gate.** None of this is admissible evidence yet. bf16 numerics on gfx1151 are untested
`[M]`/`[A]` (`ASSUMPTIONS.md → bf16-numerics-unproven`) and the Hardware Validation Gate has
not run. A ratio ablation whose published deltas are ~0.01 perplexity is precisely the
experiment unvalidated numerics would destroy. Exercise B adds a second, unrelated blocker
that the gate does not currently cover: **the timing instrument drifts.** Two identical
decode-stream runs twenty minutes apart differed by 24 % in absolute bandwidth `[M]`, which
is larger than the difference between several of the ratio arms we would want to rank. A
decode benchmark for Proteus needs arm-order randomisation, a cooldown protocol, and a
seed-to-seed null distribution before it can rank anything.

---

## 5. Read the code

All paths relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. Line numbers are pinned to the revisions in
`PROVENANCE.md`.

### 5.1 The schedule, in four implementations

| Where | What to look at, and why |
|---|---|
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:365` | `self.is_local_attention = config.layer_types[layer_idx] == "sliding_attention"`. The entire hybrid mechanism, as a list lookup at construction time. Every question in this module is a question about what goes in that list. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:366` | The ternary that sets `sliding_window` to 512 or `None`. That one line is the difference between `O(T)` and `O(w)` residency for the layer. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:343` | `def __init__(self, config, layer_idx, num_heads)` — `num_heads` arrives as a **constructor argument** sourced per layer, not read from a global config field. This is Break 4: the two tiers have different weight shapes. |
| `architecture/llama-cpp-laguna/src/models/laguna.cpp:41` | The same decision in C++ as `set_swa_pattern(4, dense_first=true)` — full attention at `il % 4 == 0`. Note the fallback: if the `sliding_window` key is absent the hybrid path is skipped entirely and the model is all-full-attention. Silent, and a 4× memory difference. |
| `architecture/samba/lit_gpt/model.py:323` | Samba stores no schedule; it recomputes `layer_idx % mb_per_layer` in the block constructor. `mb_per_layer = 2` (`architecture/samba/lit_gpt/config.py:409`) gives 1:1 Mamba:attention. |
| `architecture/samba/lit_gpt/model.py:452` | The second, independent modulo — inside the attention module — deciding windowed versus global. **No shared source of truth with `:323`.** Combined with `full_per_layer` defaulting to 1,000,000 (`architecture/samba/lit_gpt/config.py:33`), every Samba attention layer is windowed and there are **zero global layers**, which means the paper's "attention for precise recall" story covers only the last 2048 tokens. |
| `architecture/zamba2/mamba_block.py:445` | A third fusion mode most taxonomies miss. Every layer is a Mamba block; attention is not a layer type. Two shared `vBlock` modules are built with `layer_idx=-1` (`:449`) and invoked round-robin at positions flagged `'g'` in a hand-written 54-entry map, injected as a side input through a per-layer projection (`:453`). **Weight-tied attention.** |

### 5.2 The two-tier allocation, and the switch that abandons it

| Where | What to look at, and why |
|---|---|
| `architecture/llama-cpp-laguna/src/llama-kv-cache-iswa.cpp:69` | `const uint32_t size_base = kv_size;` — the expensive tier, sized to full context, filtered to non-SWA layers by the lambda just above. |
| `architecture/llama-cpp-laguna/src/llama-kv-cache-iswa.cpp:73` | `size_swa = GGML_PAD(min(size_base, n_swa·n_seq + n_ubatch), 256)` — §3.2's fixed term, allocated. This is the clearest single line of capacity planning in the reference library. |
| `architecture/llama-cpp-laguna/src/llama-kv-cache-iswa.cpp:76` | `if (swa_full)` → `size_swa = size_base`. **Break 6.** The escape hatch that promotes the cheap tier to the expensive one. There is no inverse switch anywhere in the file. |
| `architecture/llama-cpp-laguna/src/llama-graph.cpp:2891` | `const bool is_swa = hparams.is_swa(il);` — per-layer dispatch selecting which of the two cache contexts receives this layer's writes. |
| `architecture/llama-cpp-laguna/src/models/laguna.cpp:184` | `n_rot_l` — where the two tiers stop being numerically interchangeable. SWA layers get plain RoPE with YaRN's `ext_factor`, `beta_fast`, `beta_slow` all forced to zero; full layers get the long-context schedule. **Break 3.** |

### 5.3 The serving layer pays for the fixed assignment — read this section slowly

This is the most valuable read in the module, and none of it is in any paper.

| Where | What to look at, and why |
|---|---|
| `memory/vllm/vllm/v1/kv_cache_interface.py:539` | `class SlidingWindowSpec(AttentionSpec)` — a windowed layer is a *different spec type*, not a full-attention spec with a flag. Type identity is what drives grouping downstream. |
| `memory/vllm/vllm/v1/kv_cache_interface.py:584` | `num_tokens = min(self.sliding_window - 1 + max_in_flight_tokens, max_model_len)` — the admission cap for a windowed group has a **closed form** because the window is static. Hold that thought for §8's dynamic-routing item; the entire admission-control design depends on this number being computable ahead of time. |
| `memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:708` | Full attention finds a hit by scanning **from token 0 forward**, breaking on the first miss. Prefix matching. |
| `memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:918` | `# Search from right to left and early stop when a match is found.` Sliding window matches a **suffix**, needing `ceil((w−1)/block_size)` contiguous cached blocks (`:901`). Put `:708` and `:918` side by side — that asymmetry is Break 5 in two lines. |
| `memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:625` | Inside `remove_skipped_blocks`: *"A typical case is full attention that we never free any token before the request is finished."* The cheap tier's blocks are reclaimed mid-request; the expensive tier's never are. Two lifetimes, one pool. |
| `memory/vllm/vllm/v1/core/kv_cache_coordinator.py:521` | `class HybridKVCacheCoordinator` — exists solely because the tiers disagree. |
| `memory/vllm/vllm/v1/core/kv_cache_coordinator.py:633` | Full-attention groups are sorted first, because full attention is "downward-closed" and gives a tighter initial bound. That is a *scheduling* optimization over a *reconciliation* problem. |
| `memory/vllm/vllm/v1/core/kv_cache_coordinator.py:685` | `find_longest_cache_hit` — the docstring is the whole lesson: *"an iterative fixed-point algorithm. Each attention type either accepts the current candidate length or reduces it… This converges because length monotonically decreases and is bounded below by 0."* The loop is at `:727`; the termination test at `:792`. |
| `memory/vllm/vllm/v1/core/kv_cache_coordinator.py:813` | `num_uncached_common_prefix_tokens = longest_hit_length - hit_length` — the engine computes, and reports, exactly how much cache hit the hybrid structure threw away. **This is the closest thing to a hybrid miss counter that exists anywhere**, and it measures a scheduling loss, not a model loss. |
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:1140` | The grouping algorithm's docstring, which is the best plain-English account of hybrid KV management in the reference library. Read all six of its stated assumptions. |
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:1191` | *"`find_longest_cache_hit` only supports one attention type or two types of full-attention plus exactly one another type… we don't know how to implement it cleanly yet."* The general hybrid is unimplemented. |
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:1224` | `min_num_layers = min(...)` — the allocator's group size is taken from the "1" in the model's n:1 pattern, with a `FIXME` noting that this only works because every open hybrid ships an n:1 pattern. **The architecture's ratio is load-bearing inside the memory allocator.** Ship a 20-full/30-sliding model and this needs redesigning. |
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:1547` | `unify_hybrid_kv_cache_specs` — vLLM's `swa_full`. Promotes every `SlidingWindowSpec` to a `FullAttentionSpec`; the warning at `:1562` states that all KV-memory savings are surrendered while the compute saving survives. **Break 6, second implementation.** |

### 5.4 The constant-state comparison point

| Where | What to look at, and why |
|---|---|
| `architecture/mamba/mamba_ssm/modules/mamba2.py:352` | `ssm_state = torch.zeros(batch_size, self.nheads, self.headdim, self.d_state, ...)`. **No `seqlen` dimension.** Byte-identical at 1K and 1M tokens. Compare directly against `llama-kv-cache-iswa.cpp:73`, which does have a token-count term — that is the whole difference between the two "3:1"s in §3.6. |
| `architecture/mamba/mamba_ssm/modules/mamba2.py:41` | `d_state=128`, with `expand=2` at `:44` and `headdim=64` at `:45` — the three defaults §3.6's arithmetic uses. |
| `memory/vllm/vllm/v1/kv_cache_interface.py:690` | `class MambaSpec(KVCacheSpec)` — for completeness, the third spec type the coordinator has to reconcile. A Mamba group has no prefix-hit story at all in the general case. |

---

## 6. Exercises

Activate first, in PowerShell, dot-sourced so the variables survive:

```powershell
. .\scripts\activate-lab.ps1
```

**Read this before Exercise B.** Use `activate-lab.ps1` and let it set
`HIPBLASLT_TENSILE_LIBPATH` for you. If you re-derive that path in a bash driver or a CI job
and write it with forward slashes, **every GPU matmul on this machine dies with a silent
segfault** — details, evidence and the discriminating probe are in B.1, and finding out why
is worth more than the timing number that follows it.

Standing caveats also apply: single tensors ≥32 GiB hang silently at 0 % CPU `[M]`
(`ASSUMPTIONS.md → large-tensor-fault-32gib`), and bf16 numerics are untested `[M]`
(`bf16-numerics-unproven`), so accuracy claims from these exercises are provisional. Timing
claims are not affected by the numerics question; correctness claims are.

Write scratch scripts under `notebook/`. Exercise B is a Hardware Validation Gate item and
should migrate into the rig with tests on reuse.

---

### Exercise A — the crossover calculator

**Goal:** build the function that answers "which knob am I tuning?" for any hybrid config,
and use it to find a crossover between two real shipped schedules.

**Hardware:** none. Pure Python, no torch, no GPU. **Runtime:** 30 minutes to write,
under a second to run.

```python
"""Hybrid KV residency, the crossover T = r*w, and the schedule comparison."""
KIB, MIB, GIB = 1024, 1024**2, 1024**3

def bytes_per_token_per_layer(n_kv: int, head_dim: int, dtype_bytes: int) -> int:
    """One K vector and one V vector, per KV head, per token, in one layer."""
    return 2 * n_kv * head_dim * dtype_bytes

def residency(context: int, n_global: int, n_sliding: int, window: int, c: int) -> int:
    """Total KV bytes for one sequence: growing term + fixed term."""
    return c * (n_global * context + n_sliding * min(context, window))

def crossover_tokens(ratio: int, window: int) -> int:
    """Context at which the growing term equals the fixed term."""
    return ratio * window

SHIPPED = {                    # (ratio r = L_s/L_g, window w)
    "gpt-oss":       (1, 128),
    "mimo-v2-flash": (5, 128),
    "laguna-s-2.1":  (3, 512),
    "gemma-3":       (5, 1024),
    "rope-to-nope":  (3, 4096),
}
L, C = 48, bytes_per_token_per_layer(n_kv=8, head_dim=128, dtype_bytes=2)
```

**Four deliverables, each a number you can check.**

1. **`T×` and the 90 % point for every row of `SHIPPED`,** at fixed `L = 48`. Your table must
   reproduce §3.4 exactly: Laguna `T× = 1536`, 90 % of asymptote at 13,824.
   *Check:* at `T = T×` the cheap tier must hold exactly 50 % of total residency. If it does
   not, your `min(T, w)` is missing.

2. **The Laguna / gpt-oss crossover.** Solve numerically by bisection *and* by the closed
   form in §3.4. **Both must give 1280 tokens.** Then state, in one line in your notebook
   entry, which schedule you would ship for an 8k-context agent and which for a 512-token
   classifier — and note that they are different models.

3. **The saving curve, and every ordering reversal in it.** Plot `saving(T)` from §3.3 for all
   five schedules over `T ∈ [128, 1e6]` on log-x. Every curve starts at 1.0 (no saving at or
   below its own window) and asymptotes to `r+1`. Then find every pair whose *ordering
   reverses* inside the range and report the context at which each reverses.
   **Prediction: exactly four reversals** — gpt-oss↔Laguna at **1280**, gpt-oss↔Gemma 3 at
   **2368**, Laguna↔Gemma 3 at **5632**, gpt-oss↔Rope-to-Nope at **12032**. The other six
   pairs never reverse — one schedule's residency is below the other's at every context in
   range.

   **The trap, which I fell into while checking this module.** Below *both* windows of a pair,
   the two schedules are byte-identical (both are all-global), so their difference is exactly
   zero. A bisection that tests the sign of the difference at the low end of the range will
   read that zero as "no reversal" and silently drop the Laguna↔Gemma 3 pair — you will report
   three, not four. Start the search above `max(w_A, w_B)`, or handle the zero explicitly.
   The equality-point arithmetic is exact: at T = 1280, 2368, 5632 and 12032 the two
   residencies are *equal integers*, not approximately equal.

   *Consistency check on your own code:* because `saving = L·T / residency` and `L` is the
   same for all five, a saving crossing **is** a residency crossing — so the first number here
   must equal deliverable 2's 1280 exactly. If it does not, one of your two functions is
   wrong.

4. **The falsifiable one.** Assert that `residency` at `T ≤ w` equals the all-global
   residency for every schedule — i.e. **the hybrid saves literally nothing below its
   window.** If any schedule appears to save something there, you have an off-by-one in the
   `min`.

**What a wrong answer looks like.** If your Laguna asymptotic saving comes out at 3.0 rather
than 4.0, you have divided by `L_s` instead of `L`. That is the single most common error in
this whole subject and it appears in published blog posts.

---

### Exercise B — does decode time track residency across schedules?

**Goal:** test §3.5's central prediction — that wall-clock per decode step is proportional to
resident bytes, so the all-global/hybrid time ratio tracks `L / L_g` — and, before that,
reproduce a silent fault that will destroy the measurement if you build your own harness, and
practise the discipline of varying the right independent variable.

**Hardware:** one gfx1151 GPU, native Windows. **CPU fallback given below.**
**Runtime:** ~25 min on GPU — the B.1 probes are six subprocess launches at roughly 60–90 s
each because torch import dominates, and B.2 is ~4 min per run and needs at least three.
25–35 min on CPU at the reduced shapes.

#### B.1 — Reproduce the fault first, and watch a plausible explanation die

Writing B.2 killed the process with no Python traceback: exit 139, Windows exception
`0xC0000005`, top frame `hipblasLtMatmulAlgoGetHeuristic()` inside `libhipblaslt.dll`,
reached from `at::native::structured_bmm_out_cuda`. The obvious reading — *the decode-shaped
batched GEMM is broken on gfx1151* — survived four context lengths and three repeats, which
is exactly enough evidence to feel confident and be wrong. It was wrong. Working the
environment as an independent variable, instead of the shape, gave this:

`[M]` **2026-07-26, this machine, 16 crashes and 5 clean runs, every one in a fresh
subprocess. Environment:** `C:\venvs\lab`, `torch 2.12.0a0+rocm7.13.0a20260313`, HIP 7.2.0,
native Windows, gfx1151.

| `TORCH_BLAS_PREFER_HIPBLASLT` | `HIPBLASLT_TENSILE_LIBPATH` | runs | outcome |
|---|---|---|---|
| `1` | real directory, **forward slashes** (`C:/venvs/.../hipblaslt/library`) | **16** | **segfault, exit 139, no message at all** |
| `1` | same real directory, backslashes — what `activate-lab.ps1` writes | 2 | OK |
| `1` | **non-existent** directory, backslashes | 1 | clean `RuntimeError: HIPBLAS_STATUS_INVALID_VALUE`, exit 1, loud |
| `1` | unset | 1 | OK |
| `0` | real directory, forward slashes | 1 | OK |
| unset | unset | 1 | OK |

The 16 crashes span `bmm`, `mm`, 4-D `matmul` and `scaled_dot_product_attention`;
`N ∈ {512, 2048, 8192, 32768, 65536, 131072}`; `M ∈ {1, 6, 64}`; bf16 and fp32.
**Nothing about the shape matters.** It is the environment variable, and the shape sweep that
looked like evidence was sixteen re-runs of one condition.

Three things to take from this, in increasing order of importance.

**The fault, stated correctly.** A `HIPBLASLT_TENSILE_LIBPATH` written with POSIX separators
segfaults every GPU matmul on this wheel when hipBLASLt is preferred. `[A]` Mechanism, medium
confidence: the *non-existent-directory* row above fails **loudly** —
`rocblaslt error: Cannot read "…\NOSUCHDIR\TensileLibrary_lazy_gfx1151.dat"` followed by a
proper `RuntimeError` — so hipBLASLt does handle a missing library correctly. The
forward-slash path is *readable* by Windows, so it gets further into the load, and something
downstream that assumes backslashes ends up dereferencing a partially-initialised
`SolutionLibrary`. Cheapest discriminating test: a valid path with mixed separators
(`C:\venvs\lab/Lib/...`), which would separate "any forward slash" from "no backslash at all".

**The failure modes are inverted from what you want.** The obviously-wrong path fails loudly
with the offending string in the message. The subtly-wrong path — same directory, different
separator — dies silently with exit 139, no exception, no partial output, nothing in a
`nohup` log. This is the second silent-death mode this lab has found on this machine, after
the ≥32 GiB allocation hang at 0 % CPU `[M]`
(`ASSUMPTIONS.md → large-tensor-fault-32gib`), and both of them stop a long run without
producing an error. Any harness that *re-derives* this path — a bash driver, a CI job, a
`pathlib.PurePosixPath`, a config written on another platform — can hit it.
`activate-lab.ps1` writes backslashes and is correct; the hazard is in everything that
does not use it.

**It is not the row already in the register.** `ASSUMPTIONS.md → hipblaslt-config` records
that a bad Tensile path was expected to cost ~5× GEMM throughput, and that the 5× cliff was
**refuted** on this wheel. That row is about a path that still works and is merely
suboptimal. This is a different bad path with a different consequence, and the two claims are
compatible.

Your probe, one condition per process:

```python
"""Isolated probe: one matmul. Prints OK, or the process dies. Record the exit code."""
import argparse, sys, torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8192)      # cached keys
ap.add_argument("--m", type=int, default=6)         # GQA group size
ap.add_argument("--batch", type=int, default=8)     # n_kv
ap.add_argument("--k", type=int, default=128)       # head_dim
args = ap.parse_args()

a = torch.randn(args.batch, args.m, args.k, dtype=torch.bfloat16, device="cuda")
b = torch.randn(args.batch, args.n, args.k, dtype=torch.bfloat16, device="cuda")
torch.cuda.synchronize()
out = torch.bmm(a, b.transpose(1, 2))
torch.cuda.synchronize()
print(f"OK {tuple(out.shape)} finite={bool(torch.isfinite(out).all())}")
sys.exit(0)
```

Drive it with the **environment** as the independent variable — not the shape — and record
**exit codes**, because a segfault prints nothing to stdout:

```bash
LIB='C:\venvs\lab\Lib\site-packages\_rocm_sdk_libraries_gfx1151\bin\hipblaslt\library'
BAD='C:/venvs/lab/Lib/site-packages/_rocm_sdk_libraries_gfx1151/bin/hipblaslt/library'
for cfg in "1|$LIB" "1|$BAD" "1|" "0|$BAD"; do
  hbl="${cfg%%|*}"; path="${cfg#*|}"
  ( export TORCH_BLAS_PREFER_HIPBLASLT="$hbl"
    if [ -n "$path" ]; then export HIPBLASLT_TENSILE_LIBPATH="$path"; else unset HIPBLASLT_TENSILE_LIBPATH; fi
    python probe.py > /dev/null 2>&1 )
  echo "hipblaslt=$hbl path=${path:-<unset>} exit=$?"
done
```

**Deliverable — the exit-code table, and one sentence on method.** Expected: `139` for the
forward-slash row, `0` for the other three. Then write the sentence: *what would you have
concluded if you had only varied the shape?* That question, not the crash, is the exercise.
The curriculum has been here before — see `curriculum/README.md`, "Refuted: the reported
hipBLASLt segfault on skinny-K GEMMs", a crash tagged `[M]` on a basis that did not reproduce.
The fix both times is the same: **vary the thing you are not thinking about.**

**If your table differs.** If the backslash row also crashes, `activate-lab.ps1` is broken on
your wheel and every measurement taken under it needs re-examination — escalate rather than
working around it. If nothing crashes, record your wheel version; the hazard is wheel-specific
and this one is a nightly.

#### B.2 — Then the actual measurement

With the workaround in place, build a synthetic 48-layer cache whose per-layer token count
follows a hybrid schedule, and time one decode step over the whole stack.

```python
"""Does decode time track hybrid residency? Arms differ only in n_global."""
import gc, time, torch

L, N_KV, N_Q, D_H, W = 48, 8, 48, 128, 512
GIB = 1024**3

def build(tokens, dtype, dev):
    return (torch.randn(N_KV, tokens, D_H, dtype=dtype, device=dev),
            torch.randn(N_KV, tokens, D_H, dtype=dtype, device=dev))

def decode_step(cache, q, scale):
    out = None
    for k, v in cache:
        scores = torch.bmm(q, k.transpose(1, 2)) * scale
        weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.bmm(weights, v)
    return out

def arm(n_global, tokens, dtype, dev, iters, elem_bytes):
    per_layer = [tokens if i < n_global else min(tokens, W) for i in range(L)]
    cache = [build(t, dtype, dev) for t in per_layer]
    q = torch.randn(N_KV, N_Q // N_KV, D_H, dtype=dtype, device=dev)
    scale = D_H ** -0.5
    for _ in range(2):
        decode_step(cache, q, scale)
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        decode_step(cache, q, scale)
    if dev == "cuda": torch.cuda.synchronize()
    secs = (time.perf_counter() - t0) / iters
    resident = 2 * N_KV * D_H * elem_bytes * sum(per_layer)
    del cache, q; gc.collect()
    if dev == "cuda": torch.cuda.empty_cache()
    return n_global, resident / GIB, secs * 1e3, resident / secs / 1e9

dev, dtype, elem = ("cuda", torch.bfloat16, 2) if torch.cuda.is_available() else ("cpu", torch.float32, 4)
rows = [arm(g, 131072, dtype, dev, 5, elem) for g in (48, 12, 6, 4)]
base = rows[0]
for g, gib, ms, gbs in rows:
    print(f"{g:>3} global  {gib:>8.3f} GiB  {ms:>7.1f} ms  {gbs:>6.1f} GB/s  "
          f"byte_ratio={base[1]/gib:>5.2f}  time_ratio={base[2]/ms:>5.2f}")
```

**Footprint check before you run.** All-global at `T = 131072`: each layer holds
`2 × 8 × 131072 × 128 × 2 B = 512 MiB`, so 48 layers is **24 GiB** — inside the `[M]` ≥62 GiB
fast tier, and every individual tensor is 256 MiB, far below the 31 GiB hazard. Do not raise
`T` past 200k without redoing that arithmetic.

**CPU fallback.** `T = 8192`, `L = 48`, fp32, `iters = 2`. Footprint
`48 × 2 × 8 × 8192 × 128 × 4 B` = 3 GiB. The absolute bandwidth will be an order of magnitude
lower; the *ratios* are what transfer, and they are what the prediction is about.

**Deliverable — one table and one judgement.**
1. `byte_ratio` and `time_ratio` per arm. **Prediction: `time_ratio ≈ byte_ratio`**, and the
   byte ratios at `T = 131072`, `w = 512` are **1.00 / 3.95 / 7.79 / 11.51** for 48 / 12 / 6 / 4
   global layers. Note these are *below* the asymptotic `L/L_g` of 1 / 4 / 8 / 12, because
   131072 is only 256× the window and the fixed term has not quite vanished — check that
   against §3.3's `saving(T) = (r+1)T/(T + r·w)` before you look at the clock. *(Byte ratios
   are exact and computable by hand from §3.2. Do that first.)*
2. The `GB/s` column against the `[M]` 199.9 GB/s device-to-device reference. **Prediction:
   roughly flat, and below the reference** — this is a strided read over 48 separate tensors
   with a softmax between, not a pure copy.
3. **If `time_ratio` does not track `byte_ratio`, the bandwidth model under this entire track
   is wrong on this hardware.** Pre-register that as a G2 hypothesis card with SUCCESS and
   KILL thresholds before you run, not after.

**What this produced when I ran it, and why you should distrust half of it.**

`[M]` **2026-07-26, two runs, same process design, same environment** (`C:\venvs\lab`,
`torch 2.12.0a0+rocm7.13.0a20260313`, gfx1151, bf16, `T = 131072`, `iters = 5`, arms in the
order 48 → 12 → 6 → 4, resident 24.00 / 6.07 / 3.08 / 2.09 GiB):

| arm (`L_g`) | byte ratio | time ratio, run 1 | time ratio, run 2 | GB/s run 1 | GB/s run 2 |
|---|---|---|---|---|---|
| 48 (all-global) | 1.00 | 1.00 | 1.00 | 162.3 | 123.2 |
| 12 (**3:1, Laguna's**) | 3.95 | **3.78** | **3.79** | 155.0 | 118.2 |
| 6 (7:1) | 7.79 | 7.15 | 6.03 | 149.0 | 95.4 |
| 4 (11:1) | 11.51 | 10.28 | 7.52 | 145.0 | 80.5 |

**What reproduced:** the 3:1 arm, at 3.78 and 3.79 against a predicted 3.95 — within 4 %,
twice, and the residual is the fixed per-call overhead of 48 kernel launches that the byte
model does not include. `[M]` for that row, two runs.

**What did not:** everything thinner. The 11:1 arm gave 10.28 and then 7.52 — a 27 % swing
between identical runs. Absolute bandwidth fell 24 % across the whole table between run 1 and
run 2 (162 → 123 GB/s on the *same* all-global arm), which is the signature the
`uma-carveout-controls-fast-tier` notebook entry warned about: **the Z13 is a tablet and these
are sustained memory-bound loops.** Those rows are `[A]`, not `[M]`.

There is also a design flaw in the script, which is worth seeing because you will make it:
**the arms run in a fixed order, largest first.** The cheapest arms therefore always run last,
on the hottest silicon, which biases exactly the rows that degraded. Randomise or interleave
the arm order before you believe any of the thin-arm numbers, and report ≥3 runs with an
interval. This is what the house `≥3 seeds` rule is for, and this table is what it looks like
when you skip it.

**The honest conclusion.** The byte model predicts the 3:1 hybrid's decode-time advantage on
this machine to within 4 %, twice. Beyond about 8× the arms get small enough and the runs
noisy enough that a two-run experiment cannot separate the model from the thermals — which is
itself the useful finding, because it sets the effect size a Proteus decode benchmark needs
before it can distinguish ratio arms at all.

---

### Exercise C — receptive field under placement, at matched global count

**Goal:** turn §3.7 into a computed number, and produce the arm-manifest field
("reach-limited fraction") that §4 says Mnemosyne needs.

**Hardware:** none. Pure Python, no torch, no GPU. **Runtime:** 45 minutes to write, a few
seconds to run at `context = 32768` (48 × 32768 list operations).

The recurrence. Let `lo[k][t]` be the earliest source position that can influence position
`t` after `k` layers. Then `lo[0][t] = t`, a global layer gives `lo[k][t] = 0`, and a
windowed layer gives

```
lo[k][t]  =  min over s in [t-w+1, t] of lo[k-1][s]  =  lo[k-1][max(0, t-w+1)]
```

— the `min` collapses to a single lookup because `lo[k-1]` is non-decreasing in `t`. That is
`O(L·T)` and exact.

```python
"""Receptive field of a hybrid schedule, per layer, and its reach-limited fraction."""
def reach_profile(layer_types, window, context):
    """lo[k][t] = earliest source position reaching position t after k layers."""
    lo = list(range(context))
    profile = []
    for kind in layer_types:
        if kind == "full_attention":
            lo = [0] * context
        else:
            lo = [lo[max(0, t - window + 1)] for t in range(context)]
        profile.append(lo[context - 1])       # reach at the last position
    return profile

def schedule(name, n_layers=48, n_global=12):
    step = n_layers // n_global
    if name == "uniform-dense-first":  g = set(range(0, n_layers, step))
    elif name == "uniform-dense-last": g = set(range(step - 1, n_layers, step))
    elif name == "front-loaded":       g = set(range(0, n_global))
    elif name == "back-loaded":        g = set(range(n_layers - n_global, n_layers))
    elif name == "middle-loaded":      g = set(range((n_layers - n_global)//2,
                                                     (n_layers - n_global)//2 + n_global))
    else: raise ValueError(name)
    return ["full_attention" if i in g else "sliding_attention" for i in range(n_layers)]
```

**Three deliverables.**

1. **Validate the closed form.** For a pure-SWA stack (`n_global = 0`), the reach after `k`
   layers must be exactly `k·(w−1) + 1` source positions. Run `reach_profile` with
   `w = 512`, `context = 32768`, 48 sliding layers and assert
   `context - profile[k-1] == min(context, k*511 + 1)` for every `k`. **This is the check
   that can fail** — if the recurrence is off by one, this catches it immediately.

2. **The placement table.** Reproduce §3.7 for `L = 48`, `L_g = 12`, `w = 512`: for each
   schedule, report layers-below-first-global, their reach, and layers-above-last-global.
   Add `hymba-style` (globals at 0–3, 22–25, 44–47) yourself — the helper above does not
   build it, and getting it to exactly 12 globals is a good test of your indexing. Your
   `back-loaded` row must give reach **18,397**, and at `context = 32768` your reach-limited
   count must be **14,371 tokens invisible** to the bottom 36 layers.

3. **The metric that goes in the arm manifest.** Define
   `reach_limited_fraction(schedule, w, T) = mean over layers of max(0, 1 − reach_k / T)` —
   the average fraction of the context a layer cannot see. Compute it for all five schedules
   at `T ∈ {2048, 8192, 32768}` and plot. **Prediction: it is identically zero for every
   front-loaded or dense-first schedule at every `T`, and rises steeply for back-loaded as
   `T` grows.** Then answer, in one line: *does this metric distinguish the schedules that
   2510.04800 says perform differently?* If it does not, it is a necessary-but-not-sufficient
   diagnostic, and saying so precisely is worth more than the metric.

---

## 7. Self-check

Answers at the end of the file. Do not scroll.

1. A colleague says Laguna's "3:1" hybrid cuts the KV cache by 3×. At what context length is
   the saving actually exactly 3×, and what is the saving at 1024 tokens?

2. Two shipped schedules at the same depth `L = 48`: gpt-oss (1:1, w=128) and Laguna
   (3:1, w=512). Which holds less KV at 1k tokens, which at 128k, and why is there no
   context-free answer?

3. You want to test whether a wider window helps long context, so you load a pretrained
   Laguna and raise `sliding_window` from 512 to 8192 at inference time. Name the two
   independent reasons that experiment does not test what you think it tests.

4. vLLM's `HybridKVCacheCoordinator` reconciles a prefix-cache hit with a fixed-point
   iteration rather than a single pass. What property of the two managers forces the
   iteration, and what does it imply about a hybrid model's hit rate relative to an
   all-global one?

5. Both llama.cpp and vLLM ship a switch that abandons the hybrid KV saving, and neither
   ships the inverse. Why can the inverse not exist, and what does that tell you about which
   direction of re-tiering is legal at runtime?

6. You run a 3:1 versus 12:1 ratio ablation at 300M params on 0.5B tokens, score validation
   perplexity, and find no significant difference. What have you learned about the ratio?

---

## 8. What is still unsolved here

Everything below is testable at 20M–300M params on one GPU with a `[M]` ≥62 GiB fast tier,
and every item needs a pre-registered hypothesis card before it runs.

1. **Does the recall cliff below 3:1 reproduce at our scale?** 2507.06457 measured at
   340M/20B and 1.3B/100B `[C]`. A 300M/5B replication is inside budget and is the gating
   question for the entire ablation backlog — if our scale cannot see the effect,
   `ASSUMPTIONS.md → ablation-scale-sufficient` is refuted and the backlog changes shape.

2. **Contested: ceiling or schedule?** 2507.06457 finds recall collapsing below 3:1 — a
   capability ceiling. 2606.15378 finds different hybrids converging to comparable
   long-context performance given enough training — a training-speed knob `[C]`. Same year,
   same question, incompatible framings, and the resolution plausibly turns on token budget,
   which is exactly the axis a small rig can sweep. Run 3:1 and 12:1 at 0.5B and 5B tokens
   and check whether the recall gap closes. Neither paper ran it.

3. **Contested, and counterintuitive: does Large-Window Laziness appear at small scale?**
   Sweep `w ∈ {128, 512, 2048}` at fixed 3:1 and measure *when during training* long-range
   recall emerges, not just its final value `[C]` (2606.15378, 2509.24552). Cheap, and the
   mechanism claim is falsifiable. This is also the sharpest place the tiering metaphor dies,
   so a negative result would be interesting in its own right.

4. **Placement at matched global count, with a live contradiction to resolve.**
   Receptive-field arithmetic (§3.7) says front-load. 2510.04800 says never front-load, for
   Mamba hybrids at 1:12 `[C]`. Laguna front-loads `[M]`. Nemotron-H disperses evenly and
   says it inherited that `[C]`. Hymba uses first/middle/last. Compare G-S-S-S against
   front-, back- and middle-loaded at the same `L_g`, on a recall battery, and report the
   reach-limited fraction from Exercise C alongside. Every post-hoc-search paper implies this
   gap exists; none reports it at small scale with matched budgets.

5. **Nobody reports the crossover.** `T× = r·w` (§3.4) is one line of algebra and I could not
   find it in any hybrid paper. It is not deep — the claim is only that it is not reported,
   and that its absence makes cross-paper ratio comparisons incommensurable, because two
   papers evaluating at different context lengths are tuning different knobs while using the
   same word. The cheapest possible contribution in this module: report `T×` in every arm
   manifest and in every table.

6. **Does the SWA:global ratio behave like the linear:full ratio?** Both are called "3:1"
   and the literature treats them as comparable. §3.6 shows they differ by 50 % in runtime
   state at our reference shape and differ *qualitatively* in what they retain. Run both
   against the same recall battery at matched parameter **and** matched state-byte budgets.
   If they diverge, a large fraction of the cross-paper ratio comparisons in the survey note
   are invalid.

7. **Dynamic tier assignment is arriving, and it breaks capacity planning.** Switch Attention
   routes *per token, per layer* between a full-attention branch and a sliding-window branch,
   with an adaptive regularizer pushing toward efficiency, reached by continual pretraining
   from a full-attention model `[C]` (2603.26380, Mar 2026). If occupancy becomes
   data-dependent, `SlidingWindowSpec.max_admission_blocks_per_request`
   (`memory/vllm/vllm/v1/kv_cache_interface.py:584`) has no closed form — and every serving
   engine's admission control is built on that closed form existing. You are unusually well
   placed to see this one: it is the difference between provisioning for a known working set
   and provisioning for one whose size depends on the request payload — the thing that turns
   capacity planning into admission-control gambling. Nobody in the architecture literature
   has costed it, and nobody in the serving literature has implemented it.

8. **Post-hoc selection may make "what ratio?" the wrong question entirely.** FlashMorph
   formulates hybrid layer selection as budget-constrained subset optimization over frozen
   weights `[C]` (2606.30562, Jun 2026); DASH does differentiable operator allocation "in
   minutes on a single GPU" `[C]` (2605.20936, May 2026); ConSA learns FA/SWA assignment
   under a sparsity target and explicitly criticises hand-crafted rules `[C]` (2606.18056,
   Jun 2026); Jet-Nemotron's PostNAS freezes MLP weights and searches full-attention
   placement `[C]` (2508.15884, Aug 2025). The sharpest datapoint is training-free:
   NLL-guided selection reports 64.6 % on LongMemEval with **one quarter** of layers kept at
   full attention, matching a periodic **one half** baseline at 65.0 % and beating the
   periodic quarter baseline by 10.4 points, after roughly 15 minutes of one-time calibration
   `[C]` (2606.27791, Jun 2026). If a 15-minute calibration halves the full-attention budget
   at no measured cost, the interesting question stops being "what ratio" and becomes "which
   layers, under what budget" — and every one of these papers agrees that **uniform placement
   is not optimal**, which directly contradicts Nemotron-H's evenly-dispersed rule.

9. **Granularity may matter more than ratio, and the evidence for it is already better.**
   Two threads. *Fusion granularity:* 2510.04800 finds intra-layer (parallel) hybrids take the
   quality/efficiency Pareto frontier, and 2510.26912 — same month, different group — finds
   parallel hybrids more effective at long context while sequential wins at short `[C]`. Two
   independent papers pointing the same way is more support than the ratio question has, and
   **every production model in §2.5 is inter-layer.** *Head granularity:* HydraHead reports
   that heads within a layer are functionally specialized while layers exhibit block-wise
   similarity, and that a **7:1 head-wise** linear:full ratio matches a **3:1 layer-wise**
   hybrid on long context at 15B tokens `[C]` (2606.20097, Jun 2026). If either thread holds,
   the layer-ratio literature is quantizing at the wrong granularity. Both are one or two
   papers deep; a single controlled comparison at 300M would be informative for each, and the
   fusion one is the cheaper of the two to build.

10. **The attribution gap, which is the lab's actual target.** No hybrid implementation read
    for this module emits a hit rate, a miss, or any signal that the windowed layers dropped
    something needed. The closest thing that exists anywhere is vLLM's
    `num_uncached_common_prefix_tokens` (`kv_cache_coordinator.py:813`), and that measures a
    *scheduling* loss, not a model one. Mnemosyne has to construct the counters in §4 from
    scratch; there is nothing to wrap.

11. **Our instrument is not yet trustworthy for any of this, in three separate ways.** bf16
    numerics are untested `[M]` (`bf16-numerics-unproven`), and a ratio ablation whose
    published effect sizes are ~0.01 perplexity is exactly the experiment that an unvalidated
    numerics stack would destroy. Separately, Exercise B found that the *timing* instrument is
    not stable either: identical decode-stream runs 20 minutes apart differed by 24 % in
    absolute bandwidth and by 27 % in the 11:1 arm's time ratio `[M]`, consistent with thermal
    throttling on a tablet chassis. Before any decode benchmark can rank ratio arms, someone
    has to characterise that drift — arm-order randomisation, a cooldown protocol, and a
    seed-to-seed null distribution — and that work is not in the Hardware Validation Gate as
    written. `research/synthesis.md` already argues the gate is under-specified; this is one
    more item for it. Third, and smallest: the forward-slash `HIPBLASLT_TENSILE_LIBPATH`
    segfault (B.1, `[M]` 16/16) is a silent-death mode that belongs in `ASSUMPTIONS.md`
    alongside the ≥32 GiB hang.

---

## Answers to the self-check

**1.** Use `saving(T) = (r+1)T / (T + r·w)` from §3.3 with `r = 3`, `w = 512`, so `r·w = 1536`.
Set it to 3: `4T = 3T + 4608` → **T = 4608 tokens**. At `T = 1024`, saving
= `4 × 1024 / (1024 + 1536)` = 4096/2560 = **1.6×**. Check it against the raw bytes:
all-global `48 × 1024 = 49,152` token-units, hybrid `12 × 1024 + 36 × 512 = 30,720`; the
ratio is 1.6. ✓ The asymptote is `r + 1 = 4×`, reached only as `T → ∞`, and 3× is a
*mid-curve* value, not the answer. Anyone quoting "3×" is quoting the ratio and calling it a
saving.

**2.** At 1k tokens gpt-oss holds less (`24·1024 + 24·128 = 27,648` versus Laguna's
`12·1024 + 36·512 = 30,720`); at 128k Laguna holds far less
(`12·131072 + 18,432 = 1,591,296` versus `24·131072 + 3,072 = 3,148,800`). The crossover is
at exactly **1280 tokens** (§3.4). There is no context-free answer because the two schedules
trade a *slope* against an *intercept*: gpt-oss buys a tiny fixed term (24 layers × 128
tokens) at the cost of double the growing term (24 global layers instead of 12). Which one
wins is a question about your operating context, and the answer flips inside the range both
models are marketed for.

**3.** First, **the positional encoding**. Laguna's SWA layers apply plain RoPE over all 128
head dims at θ=10000 while its global layers use YaRN-scaled RoPE over 64 dims at θ=500000
(`laguna.cpp:184`) `[M]`. The sliding layers were never trained with an encoding that
reaches past 512, so widening the window puts them in a positional regime they have no
representation for. You will measure the encoding failing. Second, **the window is a
training-time variable, not an inference-time one**. Both 2509.24552 and 2606.15378 report
that window size shapes the *optimization trajectory* — short windows force the model to
train its long-range machinery; long windows delay retrieval-head formation `[C]`. Whatever
a widened window does to a finished checkpoint, it cannot tell you what training with that
window would have produced. The two reasons are independent: fixing the RoPE would not fix
the second one.

**4.** The full-attention manager matches a **prefix**, scanning left to right from token 0
and breaking on the first miss, because the block hashes are chained
(`single_type_kv_cache_manager.py:708`). The sliding-window manager matches a **suffix**,
scanning right to left for a contiguous run long enough to fill the window (`:918`). Full
attention is *downward-closed* — shortening the candidate length keeps it valid — but the
sliding-window match is **not**, because shortening the candidate can change which contiguous
run qualifies. So one group reducing the candidate can invalidate another group's already
computed answer, and the only correct resolution is to iterate to a fixed point
(`kv_cache_coordinator.py:685`, `:727`). The implication: a hybrid's hit is the
**intersection** across tiers, so its prefix-cache hit rate is bounded by the least-cacheable
group and can never exceed the all-global rate on the same traffic. The engine even quantifies
the loss (`:813`) — and that quantity has no counterpart in any storage hierarchy, where a
hit in any tier is simply a hit.

**5.** The inverse cannot exist because **promotion is always correct and demotion is never
correct.** A full-attention cache is a strict superset of a windowed one; run a windowed
layer against it and the mask discards the extra entries, so `swa_full`
(`llama-kv-cache-iswa.cpp:76`) and `unify_hybrid_kv_cache_specs`
(`kv_cache_utils.py:1547`) are safe by construction. Going the other way would mean serving a
global layer from a window-sized cache, and the bytes for the out-of-window tokens were never
written — there is no lower tier holding them, no fault path, and no way to recover them
short of a full recompute of the prefix. So the only legal runtime re-tiering move is the one
that increases memory consumption. That is the exact inverse of every storage system you have
run, where spilling down is always available and pulling up is the optimization, and it is
the clearest single statement of Break 6.

**6.** Essentially nothing about the ratio. **A flat perplexity surface is the expected
result**, not a null: Gemma 3 reported minimal perplexity impact across 1:1 to 7:1, Jamba
found 1:3 ≈ 1:7, Kimi Linear's reported spread is ~0.01–0.05 PPL across a 15× span of
ratios, and 2507.06457 found language modelling stable across ratios in a 72-model sweep
`[C]`. You have reproduced the field's consensus that the ratio does not move perplexity;
you have not learned whether it moves capability. The metric on which the surface is *not*
flat is recall, and you did not score it. Worse, at 0.5B tokens you are sitting exactly on
the contested axis: 2507.06457 predicts a recall gap (a ceiling), 2606.15378 predicts the gap
closes with more training (a schedule) `[C]`. Re-run with an MQAR/RULER-style probe and two
token budgets and the same compute buys you an answer to a live disagreement instead of a
confirmation of a known null.

---

## Sources

**Local artifacts and measurements (`[M]`)**

- `research/reference/models/laguna-s/config.json` at revision `b0a9fd7c850e`
  (`research/reference/PROVENANCE.md`), re-read 2026-07-26: `num_hidden_layers` 48,
  `num_key_value_heads` 8, `head_dim` 128, `sliding_window` 512, `hidden_size` 3072,
  `max_position_embeddings` 1048576, `torch_dtype` bfloat16, `layer_types` = 12
  `full_attention` + 36 `sliding_attention` in a strict GSSS pattern,
  `num_attention_heads_per_layer` = 48 on the full layers and 72 on the sliding ones.
- `ASSUMPTIONS.md` rows: `reference-model`, `kv-per-token-laguna`, `laguna-heads-uniform`,
  `gpu-fast-tier-size`, `large-tensor-fault-32gib`, `bf16-numerics-unproven`,
  `ablation-scale-sufficient`, `hipblaslt-config`, `torch-build`, `single-device-only`.
- `notebook/uma-carveout-controls-fast-tier.md` — ~200 GB/s flat to ≥62 GiB, single run per
  arm, 2026-07-26.
- Exercise B.1, the forward-slash `HIPBLASLT_TENSILE_LIBPATH` segfault — measured on this
  machine 2026-07-26: 16 crashes and 5 clean runs, every one in a fresh subprocess, full
  condition table and environment in §6. Not yet an `ASSUMPTIONS.md` row; it should become
  one, and it is a Hardware Validation Gate item.
- Exercise B.2, the hybrid decode-stream timing — measured on this machine 2026-07-26, two
  runs, table and caveats in §6. The 3:1 row (3.78 / 3.79 against a predicted 3.95) is the
  only row that reproduced; the thinner arms drifted with what looks like thermal throttling
  and are `[A]`, not `[M]`.
- Every `file:line` in §5 was opened and the named construct confirmed on the named line on
  2026-07-26, against the revisions in `PROVENANCE.md`.

**Repo documents this module builds on**

- `research/memory/hybrid-architectures.md` — the survey this module teaches. Nothing here
  contradicts it; §2.5 adds a Rope-to-Nope row to its ledger and §2.6(c) supplies the
  2510.04800 conclusions it flagged as unextracted.
- `curriculum/attention-variants-and-kv-cost.md` — the KV product, GQA, and the decode
  arithmetic-intensity derivation, all assumed here.
- `research/memory/memory-failure-register.md` (`hybrid-ratio-sensitivity`),
  `research/memory/kv-serving-hierarchy.md`, `research/synthesis.md`,
  `research/reference/CODE_MAP.md`.

**arXiv (`[C]`)**

- `2306.14048` — *H2O: Heavy-Hitter Oracle for Efficient Generative Inference* (2023).
- `2312.00752` — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023).
- `2312.04927` — *Zoology: Measuring and Improving Recall in Efficient Language Models* (2023).
- `2402.18668` — *Simple linear attention language models balance the recall-throughput
  tradeoff* (2024).
- `2403.17844` — *Mechanistic Design and Scaling of Hybrid Architectures* (2024-03-26).
- `2403.19887` — *Jamba: A Hybrid Transformer-Mamba Language Model* (2024-03).
- `2404.06654` — *RULER: What's the Real Context Size of Your Long-Context Language Models?* (2024).
- `2404.14469` — *SnapKV: LLM Knows What You are Looking for Before Generation* (2024).
- `2405.21060` — *Transformers are SSMs: Generalized Models and Efficient Algorithms Through
  SSD* (2024).
- `2406.07887` — *An Empirical Study of Mamba-based Language Models* (Waleffe et al., 2024).
- `2411.13676` — *Hymba: A Hybrid-head Architecture for Small Language Models* (2024-11-20).
- `2501.18795` — *Rope to Nope and Back Again: A New Hybrid Attention Strategy* (2025-01-30).
  Ratio ablation over 1:1 / 1:3 / 1:7 with 1:3 chosen; NoPE on global layers, RoPE with
  w=4096 on local. Verified by fetching the arXiv abstract and HTML pages 2026-07-26.
- `2503.19786` — *Gemma 3 Technical Report* (2025-03-25).
- `2504.03624` — *Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer
  Models* (2025-04).
- `2506.15545` — *RATTENTION: Towards the Minimal Sliding Window Size in Local-Global
  Attention Models* (2025-06-18). Reports a 512-token window matching full attention when the
  out-of-window residual is carried by a linear-attention term. Verified 2026-07-26.
- `2507.06457` — *A Systematic Analysis of Hybrid Linear Attention* (2025-07-08, rev.
  2026-06-24). The 72-model sweep; recall collapse below 3:1.
- `2508.10925` — *gpt-oss-120b & gpt-oss-20b Model Card* (2025-08-08).
- `2508.15884` — *Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search*
  (2025-08-21).
- `2509.24552` — *Short window attention enables long-term memorization* (2025-09-29, rev.
  2026-05-04). Stochastic window size.
- `2510.04800` — *Hybrid Architectures for Language Models: Systematic Analysis and Design
  Insights* (2025-10-06, rev. 2026-04-21). Placement and ratio recipes read from the arXiv
  HTML render 2026-07-26.
- `2510.19338` — *Every Attention Matters: An Efficient Hybrid Architecture for Long-Context
  Reasoning* (Ring-linear, 2025-10-22).
- `2510.26692` — *Kimi Linear: An Expressive, Efficient Attention Architecture* (2025-10-30).
- `2510.26912` — *Understanding and Enhancing Mamba-Transformer Hybrids for Memory Recall and
  Language Modeling* (2025-10-30). Sequential versus parallel fusion; sequential better at
  short context, parallel better at long. Verified by fetching the arXiv abstract page
  2026-07-26 — and see §2.5 point 3 for a correction to how our own register cites it.
- `2601.02780` — *MiMo-V2-Flash Technical Report* (2026-01-06).
- `2603.26380` — *Switch Attention: Towards Dynamic and Fine-grained Hybrid Transformers*
  (2026-03). Per-token, per-layer routing between full and windowed branches. Verified
  2026-07-26.
- `2605.20936` — *DASH: Fast Differentiable Architecture Search for Hybrid Attention in
  Minutes on a Single GPU* (2026-05-20).
- `2606.15378` — *Rethinking the Role of Efficient Attention in Hybrid Architectures*
  (2026-06-13). Large-Window Laziness; the convergence framing.
- `2606.18056` — *ConSA: Controllable Sparsity in Hybrid Attention via Learnable Allocation*
  (2026-06-16).
- `2606.20097` — *HydraHead: From Head-Level Functional Heterogeneity to Specialized
  Attention Hybridization* (2026-06-18).
- `2606.27791` — *NLL-Guided Full-Attention Layer Selection for Training-Free Sliding-Window
  Adaptation* (2026-06-26). 64.6 % on LongMemEval at 1/4 full-attention layers versus 65.0 %
  for a 1/2 periodic baseline; ~15 min calibration. Verified 2026-07-26.
- `2606.30562` — *Morphing into Hybrid Attention Models* (FlashMorph, 2026-06-29).

Every id above other than the six marked as page-verified this session (2501.18795,
2506.15545, 2510.26912, 2603.26380, 2606.27791, and the 2510.04800 HTML read) appears in
`research/memory/citation-verification.json` or `research/notes/citation-verification.json`,
both resolved against the live arXiv API.
