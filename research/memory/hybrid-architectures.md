---
title: Hybrid architectures — inter-layer, head-wise, and where the ratios actually come from
version: 1.0.0
track: research/memory
written: 2026-07-26
---

# Hybrid architectures: inter-layer, head-wise, and where the ratios actually come from

**What this note settles.** The hybrid design space has exactly two structural axes — *which
cheap primitive* you substitute (sliding-window softmax attention, or a constant-state
recurrence) and *at what granularity* you fuse it with full attention (whole layers, or heads
inside a layer) — and almost every shipped model in 2026 is one point on that grid. On the
central question, the honest answer is **mixed and mostly unflattering**: four teams ran a real
ratio ablation and published it, at least four more inherited a number by citation or by
precedent and said so, and the ablations that do exist show a *flat* loss landscape in the ratio
with a *sharp* recall landscape — which means perplexity-selected ratios are selected on noise.
The strongest 2026 finding is not about the ratio at all: window size and ratio interact in the
opposite direction from intuition, and both control how *fast* long-context ability appears more
than whether it appears.

---

## 1. Fixing the vocabulary, because "3:1" is four different numbers

Four independent knobs get collapsed into one phrase in the literature. Separate them before
comparing anything.

| Knob | What it is | Values in the wild |
|---|---|---|
| **Efficient primitive** | What replaces full attention in the cheap layers | Sliding-window softmax attention (SWA); a linear-attention / SSM recurrence (Mamba-2, Gated DeltaNet, KDA, xLSTM); block-sparse attention |
| **Fusion granularity** | Where the two primitives meet | **Inter-layer** (a layer is one type or the other); **intra-layer / head-wise** (both types run in the same layer, on the same input); **weight-tied** (one shared attention module invoked on a schedule) |
| **Ratio** | How many cheap units per expensive unit | 1:1 to 12:1, with 3:1 the current fashion |
| **Window / state size** | The cheap layer's memory budget | SWA windows 128–4096 tokens; recurrent states of a few hundred KB per layer |

Two traps. First, **the denominator moves.** Jamba's "1:7" is attention:Mamba; Kimi Linear's
"3:1" is linear:full; Gemma 3's "5:1" is local:global. Same colon, different fraction. Second,
**the primitive changes what the ratio means.** A 3:1 SWA:global model still has softmax
attention everywhere — the cheap layers just have a bounded receptive field, and their KV cost is
`O(window)` rather than `O(0)`. A 3:1 linear:full model has *no* attention in three of four
layers and a genuinely constant-size state. These are not the same architecture and there is no
reason their optimal ratios should coincide.

### The arithmetic, symbol by symbol

For a pure-attention decoder the KV cache is

```
bytes  =  2 · L · H_kv · d_head · b · T
```

- `2` — one **K**ey vector and one **V**alue vector stored per token per head.
- `L` — number of layers. Every layer keeps its own cache; they are not shared.
- `H_kv` — key/value heads per layer. Under GQA `[C]` (2305.13245, 2023) this is smaller than
  the query-head count; it is the only head number that affects cache size.
- `d_head` — dimensions per head (128 is standard).
- `b` — bytes per stored scalar (2 for bf16).
- `T` — tokens currently in context. The only term that grows at run time.

Make it hybrid: global layers keep the `T` term, windowed layers cap at the window `w`, and
recurrent layers drop out of the sum entirely and contribute a constant instead:

```
bytes  =  2 · b · d_head · [  Σ_(global l) H_kv,l · T  +  Σ_(local l) H_kv,l · min(T, w)  ]
          +  Σ_(recurrent l) H_l · d_k · d_v · b
```

The last term — a recurrent layer's state — is `H_l` heads times a `d_k × d_v` matrix per head,
with **no `T` in it at all** `[C]` (2312.00752, 2023; 2405.21060, 2024).

**Worked on the reference model.** Laguna S 2.1: 48 layers, 12 `full_attention` + 36
`sliding_attention` in a strict G-S-S-S period-4 pattern, `sliding_window` 512, `H_kv` 8,
`d_head` 128 `[M]` (config fetched at `b0a9fd7c850e`, `ASSUMPTIONS.md → reference-model`;
mechanism at `modeling_laguna.py:365-366` and `laguna.cpp:41`, `CODE_MAP.md`).

- If every layer were global: `2·48·8·128·2 B = 192 KiB/token` → **24 GiB at 128k context**
  `[M]` (`ASSUMPTIONS.md → kv-per-token-laguna`).
- Actual: global part `2·12·8·128·2 B = 48 KiB/token` → 6.0 GiB at 128k. Local part is *capped*:
  `2·36·8·128·2 B × 512 tokens = 72 MiB`, **constant** past 512 tokens. Total ≈ **6.07 GiB**.

So the useful closed form: as `T → ∞` the KV reduction from an inter-layer SWA hybrid converges
to the **period** of the pattern, `L / L_global` — here exactly 4×. Not the ratio, the ratio plus
one. Below `T = w` the hybrid saves nothing at all; the whole benefit is asymptotic.

**Caveat, and it is ours:** Laguna sources per-layer query-head counts from
`config.num_attention_heads_per_layer`, not the top-level `num_attention_heads: 48`
`[M]` (`modeling_laguna.py:343`; `ASSUMPTIONS.md → laguna-heads-uniform`, status **refuted**).
Query heads do not enter the KV formula, and `num_key_value_heads` is uniform at 8 with no
per-layer override `[M]` (verified against `config.json` at `b0a9fd7c850e`, 2026-07-26), so
**192 KiB/token is exact**. What varies per layer is the GQA group size `G = H_q/H_kv`:
**6 on the 12 full-attention layers, 9 on the 36 sliding ones** — which changes decode
arithmetic intensity, not KV bytes.

**What that costs on our instrument.** Our measured fast memory tier sustains ~200 GB/s out to
≥62 GiB `[M]` (`notebook/uma-carveout-controls-fast-tier.md`, single run per arm, 2026-07-26).
Decode must stream the whole KV cache once per token. 6.07 GiB ÷ 200 GB/s ≈ **33 ms/token**
(≈31 tok/s) for the hybrid; 24 GiB ÷ 200 GB/s ≈ **129 ms/token** (≈8 tok/s) if it were dense.
That is arithmetic over two measured inputs, not a benchmark — it ignores weight traffic, which
for a 118B-A8.5B MoE is itself ~17 GB/token at bf16 and would dominate until context passes
~100k. The systems point survives anyway: **the ratio is a bandwidth decision before it is a
quality decision**, and at long context it is the dominant one.

---

## 2. Inter-layer hybrids: the shipped design

The mechanism is embarrassingly simple, and reading it is worth more than any diagram.

- **Laguna** looks the layer type up in a list: `self.is_local_attention = config.layer_types[layer_idx]`
  `[M]` (`modeling_laguna.py:365`). The entire hybrid-ratio research question is *what goes in
  that list*, which makes it trivially ablatable.
- **Samba** does not store a schedule at all; it recomputes one from `layer_idx % mb_per_layer`
  inside each block's constructor, the way a striping function derives a device from a block
  number `[M]` (`samba/lit_gpt/model.py:323`). `mb_per_layer = 2` gives a **1:1** Mamba:attention
  ratio `[M]` (`config.py:409`) — and `full_per_layer` defaults to 1,000,000, so **every** Samba
  attention layer is windowed and there are **zero global layers** `[M]` (`model.py:452`,
  `config.py:33`). The paper's "attention for precise recall" story therefore only covers the
  last 2048 tokens; everything older is carried by the Mamba state alone.
- **Zamba2** is structurally different, not just differently-ratioed: every layer is a Mamba
  block, and attention is not a layer type. Two shared `vBlock` modules are built with
  `layer_idx=-1` and invoked round-robin at 9 positions flagged `'g'` in a hand-written 54-entry
  layer map, injected as a *side input* through a per-layer low-rank projection `[M]`
  (`mamba_block.py:445-453`, read this session, `CODE_MAP.md`). That is **weight-tied attention**
  — a third fusion mode the two-axis taxonomies mostly miss `[C]` (2411.15242, Nov 2024).
- **Jamba** ships 32 layers in 4 blocks of 8, **1 attention : 7 Mamba**, 4 attention layers total
  `[C]` (2403.19887, Mar 2024).
- **Nemotron-H** ships 8B as 4 attention / 24 Mamba-2 / 24 FFN out of 52 layers, and 56B as
  10 / 54 / 54 out of 118 `[C]` (2504.03624, Apr 2025).

### The systems bridge, and the three places it breaks

llama.cpp's Laguna path allocates **two entirely separate KV caches** — a full-size one holding
only the global layers and a small one sized to `n_swa + n_ubatch` `[M]`
(`llama-kv-cache-iswa.cpp:73`). That reads exactly like a two-tier storage hierarchy with
capacity planning. It is not one, in three specific ways, and the breaks are the teaching part:

1. **No promotion, demotion, or miss path.** A layer is bound to one tier forever by its index,
   decided from a config key at load time. Nothing migrates.
2. **"Eviction" from the small tier is not a policy with a hit-rate cost.** `is_masked_swa` makes
   out-of-window tokens *architecturally unreadable*, so discarding them is lossless, not a
   gamble. Contrast an H2O/SnapKV eviction policy, where discarding is a bet `[C]` (2306.14048,
   2023; 2404.14469, 2024).
3. **The tiers are not numerically interchangeable.** Global layers apply YaRN-scaled RoPE over
   64 of 128 head dims at θ=500000; SWA layers apply plain RoPE over all 128 at θ=10000 `[M]`
   (`laguna.cpp:184`). You **cannot** "just widen the windows" to test long context — the SWA
   layers were never trained with positional encoding that reaches past the window.

The deepest break, and the one that motivates Mnemosyne's attribution instrumentation: **a
hybrid never knows it missed.** There is no hit-rate counter, no fault, no error return. A cache
that silently returns wrong data on a miss is not a cache; it is an approximation with no
observability surface. Every measurement in this track has to be constructed, because the
architecture emits nothing.

---

## 3. Intra-layer and head-wise fusion

The alternative is to run both primitives *in the same layer, on the same input*, and sum their
outputs. **Hymba** does this: attention heads and SSM heads process the identical projected
input in parallel, and the outputs are combined as `β₁·norm(M_attn·X) + β₂·norm(M_ssm·X)` with
`β₁, β₂` learnable per-channel rescalings — necessary because SSM head outputs are consistently
larger in magnitude `[C]` (2411.13676, Nov 2024). Note that Hymba is *also* an inter-layer
hybrid: only 3 of its layers (first, middle, last) use global attention, the rest are windowed,
and consecutive layers share one KV cache. Stacked, those give 11.67× cache reduction versus
Llama-3.2-3B (918 MB → 79 MB at 8k, fp16).

**Systems analogy:** two replicas serving the same read, one exact and slow, one approximate and
cheap. **Where it breaks:** you do not *choose* a replica and you cannot fall back — the outputs
are summed unconditionally, so you always pay for both, and a bad approximate replica actively
corrupts the answer rather than merely being unhelpful. There is no quorum and no read repair.

**Why head granularity is defensible.** HydraHead's interpretability analysis reports that
*layers* exhibit block-wise functional similarity while *heads within a layer* are functionally
specialized despite sharing input features `[C]` (2606.20097, Jun 2026). If true, layer-wise
allocation is quantizing at the wrong granularity. Their headline: with retrieval-critical heads
identified and kept as full attention, a **7:1** head-wise linear:full ratio matches a **3:1**
layer-wise hybrid's long-context performance, trained on only 15B tokens. That is the sharpest
existing argument that the whole layer-ratio literature is measuring an artifact of the chosen
granularity. It is one paper, one month old, and unreplicated.

The systematic comparison of the two axes is 2510.04800 (Oct 2025, rev. Apr 2026) `[C]`, which
evaluates inter-layer (sequential) against intra-layer (parallel) fusion on quality, long
context, scaling, and train/inference cost, and proposes design recipes. I could not extract its
per-axis numeric conclusions from the abstract page this session; read the paper before citing a
winner.

---

## 4. The ledger: ablated, fitted, or inherited

This is the question the note exists to answer. Classification is by **what the source document
itself says**, not by what a later summary claims for it.

| Model / study | Ratio (as shipped) | How the number was chosen | Evidence class |
|---|---|---|---|
| Jamba `[C]` 2403.19887 | 1 attn : 7 Mamba | Compared pure-attention, pure-Mamba, 1:3 and 1:7 at **1.3B/250B tokens** and **7B/50B tokens**; 1:3 and 1:7 performed similarly, 1:7 chosen as more compute-efficient | **Ablated** (and the ablation says the ratio barely mattered) |
| Gemma 3 `[C]` 2503.19786 | 5 local : 1 global, w=1024 | Swept local:global (incl. 1:1, 7:1) and window sizes (1024, 4096); reported *minimal* perplexity impact across the sweep; KV overhead 60% (global-only) → <15% (1:3, w=1024) | **Ablated**, and the ablation reports a flat quality surface — the ratio was chosen on memory, not loss |
| Kimi Linear `[C]` 2510.26692 | 3 KDA : 1 MLA, NoPE on the MLA layers | Ratio ablation over 0:1 / 1:1 / 3:1 / 7:1 / 15:1; 3:1 best | **Ablated** — see the caveat below |
| Ring-linear `[C]` 2510.19338 | 4:1 (16B model), 7:1 (104B model) | Scaling-law curves fitted per layer-group size M, then "balance efficiency and effectiveness" | **Fitted** — and notably *the same team chose different ratios at different scales* |
| Hybrid-linear systematic study `[C]` 2507.06457 (rev. Jun 2026) | recommends 3:1–6:1 | **72 models** open-sourced: 36 at 340M/20B tokens, 36 at 1.3B/100B tokens, 6 linear variants × 5 ratios | **Ablated** — the only large controlled academic sweep in the literature |
| MAD `[C]` 2403.17844 | hybrid topology, no single ratio published in the abstract | 500+ models, 70M–7B, synthetic unit tests shown predictive of compute-optimal perplexity | **Ablated** (topology search; ratio not the headline) |
| Hymba `[C]` 2411.13676 | 3 global of 32 layers; parallel heads | Table 1 gives a seven-step A→G roadmap with a measured delta per architectural change | **Ablated**, incrementally and attributably |
| **Nemotron-H** `[C]` 2504.03624 | ~8% attention layers, evenly dispersed | The paper states it set attention to roughly 8% of layers *as suggested by prior work*, citing Waleffe et al. `[C]` 2406.07887 | **Inherited, and says so** |
| **gpt-oss** `[C]` 2508.10925 | 1:1 alternating banded (w=128) / dense | The model card attributes the alternating pattern to GPT-3 precedent; no ablation given | **Inherited, and says so** |
| **MiMo-V2-Flash** `[C]` 2601.02780 (Jan 2026) | 5:1 SWA:global, w=128 | No ratio ablation in the report | **Undisclosed** |
| **Qwen3-Next / Qwen3.5** `[C]` (model cards; Qwen3.5 released 2026-02-16) | 3 Gated DeltaNet : 1 full | No public ratio ablation found | **Undisclosed** |
| **Laguna S 2.1** `[M]` (our fetch) | 3 SWA : 1 global, w=512 | There is no paper. The ratio is readable only from the shipped artifact | **Undisclosed** |

**The Kimi Linear caveat, stated plainly.** Secondary summaries of its ablation table report
validation perplexity of 5.65 (3:1), 5.66 (1:1), 5.70 (7:1), 5.77 (full attention), 5.82 (15:1).
**I could not fetch the primary table this session** — the arXiv HTML renders with a fatal error
and the PDF was not extractable in this environment. Treat those five numbers as second-hand and
verify before quoting. If they are right, they make the central point better than any argument
could: the winning ratio beats 1:1 by **0.01 perplexity**, and a study that declared 1:1 the
winner would have been equally defensible. That is a flat surface being read as a peak.

### What the ledger actually shows

1. **Real ablations exist.** The claim that hybrid ratios are pure folklore is false. Jamba,
   Gemma 3, Kimi Linear, Hymba, Ring-linear, MAD and the 72-model sweep are all real work.
2. **Every ablation that reports a quality surface reports a flat one.** Gemma 3: minimal
   perplexity impact from 1:1 through 7:1. Jamba: 1:3 ≈ 1:7. Kimi Linear: ~0.01–0.05 PPL across
   a 15× span of ratios. 2507.06457: language modelling stable across ratios. When the objective
   is flat, the selected point is determined by whatever tie-breaker the team used — usually
   memory or throughput — and *that is fine*, but it is not the same claim as "3:1 is optimal."
3. **Recall is where the surface is not flat.** 2507.06457's core finding: language modelling is
   stable across the linear:full ratio, but **recall degrades sharply once full-attention layers
   thin below 3:1** `[C]` (2507.06457, rev. Jun 2026). Perplexity-selected ratios are therefore
   selected on the one metric that cannot see the failure. This is the same category error the
   KV-eviction literature keeps making.
4. **Frontier ratios are largely inherited, and the honest ones admit it.** Nemotron-H cites its
   source. gpt-oss cites GPT-3. That is not folklore — it is citation. The folklore risk is
   downstream: a number justified once at one scale with one primitive gets carried across
   scales and primitives where nothing re-derives it.
5. **The one datapoint suggesting scale-dependence:** Ring-linear ships 4:1 at 16B and 7:1 at
   104B from the same recipe and the same scaling-law procedure `[C]` (2510.19338). If ratio
   genuinely scales with model size, every fixed-ratio claim is scale-local, and a small-scale
   ablation rig must report the scale it measured at, loudly.

---

## 5. The three findings that are more robust than the ratio

**(a) Hybrid beats either pure at matched budget.** Jamba at 1.3B and 7B `[C]` (2403.19887);
MAD across 500+ models 70M–7B `[C]` (2403.17844); 2507.06457 across 72 models `[C]`. This is the
best-supported claim in the track.

**(b) Longer windows *hurt* long context.** Two independent 2026 results, from different groups
and different primitives, agree:
- SWAX (SWA + xLSTM): larger sliding windows hurt long-context performance, because short
  windows *force* the model to train its long-term memory rather than lean on local softmax.
  Also validated on SWA+full-attention hybrids. Too-small windows hurt short-context tasks, so
  they train with a **stochastic** window size and beat fixed windows on both `[C]` (2509.24552,
  Sep 2025, rev. May 2026).
- **"Large-Window Laziness":** larger SWA windows *delay the formation of retrieval heads* in the
  full-attention layers. Long-range retrieval is carried by full attention; the efficient module
  shapes its optimization trajectory `[C]` (2606.15378, Jun 2026).

For a caching engineer this is the counterintuitive one: **increasing the fast tier's capacity
degrades the slow tier's behaviour**, because the two tiers are not independent — they are
co-trained, and the cheap one crowds out the expensive one's learning signal. There is no
storage-hierarchy analogue. It is the clearest place where the tiering metaphor fails.

**(c) The ratio may be a training-speed knob, not a capability ceiling.** 2606.15378's scaling
analysis finds different hybrids **converge to comparable long-context performance given enough
training**, with the efficient-attention design controlling how fast that capability emerges
`[C]`. Its constructive result: applying NoPE to only the full-attention layers of a
small-window SWA hybrid substantially improves long context at negligible short-context cost —
which is, note, exactly what Kimi Linear ships (NoPE on all MLA layers) `[C]` (2510.26692).

---

## 6. Contested — present as contested

**Ceiling or schedule?** 2507.06457 finds recall collapses below 3:1, implying a real capability
ceiling. 2606.15378 finds convergence to comparable performance under sufficient training,
implying a training-efficiency knob. Same year, same question, incompatible framings. The
resolution plausibly depends on token budget — which is precisely the axis a small rig can
attack, since 2507.06457's arms were 20B and 100B tokens and the convergence claim is about the
long-budget limit.

**Does hybridization survive at frontier scale?** MiniMax abandoned hybrid Lightning Attention
for M2, stating that hybrids looked equal to full attention on MMLU/BBH/MATH/LongBench but showed
clear deficits in multi-hop reasoning at scale, that an SWA variant degraded as context grew, and
that linear-attention infrastructure (low-precision state storage, prefix caching, speculative
decoding) is immature `[C]` (MiniMax, "Why Did M2 End Up as a Full Attention Model?",
2025-10-30; series report 2605.26494, May 2026). MiniMax M2.5 (2026-02-11) ships plain MHA on
reliability grounds `[C]`. Kimi Linear claims the opposite under matched-scale pretraining
`[C]` (2510.26692). Neither is a controlled academic ablation; both are shipping-product
retrospectives with commercial incentives. MiniMax has since teased sparse rather than linear
attention `[C]` (2606.13392, Jun 2026) — a third position. As of Feb 2026 four major labs shipped
four different answers in five weeks: Qwen3.5 hybrid GDN 3:1, Kimi K2.5 MLA, GLM-5 MLA+sparse,
MiniMax M2.5 plain MHA `[C]`.

**Layer-wise vs head-wise granularity.** HydraHead argues heads are the right unit and that
7:1 head-wise ≈ 3:1 layer-wise `[C]` (2606.20097). Every production model listed above is
layer-wise. Unresolved, and one-paper-deep.

**Is the ratio even a pretraining commitment?** A 2026 wave says no: pick it *after* pretraining.
FlashMorph formulates hybrid layer selection as budget-constrained subset optimization, freezes
weights, learns layerwise gates on synthetic long-context retrieval data, then discretizes under
a full-attention budget `[C]` (2606.30562, Jun 2026). DASH does differentiable layer-wise
operator allocation "in minutes on a single GPU" `[C]` (2605.20936, May 2026). ConSA learns
FA/SWA assignment under a user-specified sparsity target, explicitly criticizing hand-crafted
rules `[C]` (2606.18056, Jun 2026). Jet-Nemotron's PostNAS freezes MLP weights and searches
full-attention layer placement `[C]` (2508.15884, Aug 2025). HALO converts Qwen3 into a hybrid
with 2.3B tokens `[C]` (2601.22156, Jan 2026). If this line holds, "what ratio?" is the wrong
question and "which layers, under what budget?" is the right one — and the answer becomes cheap
to search. Note what all of these agree on: **uniform placement is not optimal**, which
contradicts Nemotron-H's evenly-dispersed rule.

**Uniform vs non-uniform placement.** Nemotron-H disperses evenly; Hymba puts global attention at
first/middle/last; Jamba uses one attention layer per 8-layer block; the entire post-hoc-search
literature exists because placement matters. No consensus.

---

## 7. Consequences for Proteus and Mnemosyne

- **The config surface is the experimental surface, literally.** Laguna's hybrid schedule is a
  list indexed by layer number `[M]` (`modeling_laguna.py:365`). Proteus should expose
  `layer_types` as an explicit list, not a modulo — Samba's modulo-in-two-places design has *no
  shared source of truth* for the layer type `[M]` (`model.py:323` and `model.py:452`), which is
  exactly the bug shape that makes an ablation unreproducible. Samba's own escape hatch is an
  explicit position list `[M]` (`model.py:321`); start there.
- **Never report a hybrid-ratio result on perplexity alone.** Every ablation above that used
  perplexity found a flat surface. Score recall explicitly: MQAR-style multi-query associative
  recall `[C]` (2312.04927, 2023) and a RULER-style synthetic battery `[C]` (2404.06654, 2024).
- **Match state size, not just parameters.** A 3:1 linear hybrid and a 3:1 SWA hybrid at equal
  parameters have wildly different runtime memory. The recall-vs-state-size Pareto frontier is
  the real axis `[C]` (2402.18668, 2024).
- **Instrument for attribution.** The architecture emits no hit rate. If Mnemosyne is going to
  claim a mechanism, it has to construct the counter — per-layer attention mass reaching beyond
  the window, retrieval-head formation over training steps, recall as a function of distance.
  "Large-Window Laziness" was only findable because someone measured *when retrieval heads
  formed*, not *whether the model was good*.
- **Our hardware suits this question unusually well.** 62 GiB of ~200 GB/s memory `[M]` means we
  can hold KV caches that discrete cards cannot, and the platform's low bandwidth-to-compute
  ratio *magnifies* the decode bottleneck the ratio controls. Constraint: single tensors ≥32 GiB
  hang `[M]` (`ASSUMPTIONS.md → large-tensor-fault-32gib`), so KV caches must be allocated in
  sub-32 GiB chunks — which paged/blocked allocation gives for free.
- **Gate:** none of this is measurable yet. bf16 numerics are unproven on gfx1151
  `[M]`/`[A]` (`ASSUMPTIONS.md → bf16-numerics-unproven`, status untested) and the Hardware
  Validation Gate has not run. A ratio ablation whose deltas are ~0.01 perplexity is exactly the
  experiment that unvalidated numerics would destroy.

**Decision:** treat hybrid ratio as an *open* axis in Proteus, defaulting to Laguna's verified
3:1 G-S-S-S with `w=512` for arm naming continuity — and treat *window size* and *layer
placement* as first-class ablation axes alongside it, because the 2026 evidence says they matter
at least as much.
**Riskiest assumption:** that a 20M–300M ablation can resolve a difference that is ~0.01
perplexity at 1.3B–48B scale. It very likely cannot, on perplexity. It plausibly can on recall,
where the reported effect sizes are large.
**Next test:** a matched-parameter, matched-token, ≥3-seed sweep of `layer_types` at fixed window
`w=512`, scored on MQAR recall depth rather than validation loss, with the SUCCESS threshold
stated as a recall delta before any arm runs.

---

## Open questions

Testable at 20M–300M params, single GPU, ≥62 GiB fast memory tier `[M]`, no multi-GPU.

1. **Does the recall cliff below 3:1 reproduce at 20M–300M?** 2507.06457 measured at 340M/20B and
   1.3B/100B. A 300M/5B replication is within budget and would tell us whether our scale can see
   the effect at all — which is the gating question for the entire ablation backlog
   (`ASSUMPTIONS.md → ablation-scale-sufficient`, untested).
2. **Ceiling or schedule?** Run 3:1 and 12:1 at a *short* and a *long* token budget (e.g. 0.5B
   and 5B) and check whether the recall gap closes with training. That is a direct head-to-head
   between 2507.06457 and 2606.15378 and neither paper ran it.
3. **Does Large-Window Laziness appear at small scale?** Sweep `w ∈ {128, 512, 2048}` at fixed
   3:1 and measure *when during training* long-range recall emerges, not just its final value.
   Cheap, and the mechanism claim is falsifiable.
4. **Is uniform placement worse than searched placement at fixed budget?** Compare G-S-S-S
   against front-loaded, back-loaded, and Hymba-style first/middle/last, with the same count of
   global layers. Every post-hoc-search paper implies this gap exists; none reports it at small
   scale with matched budgets.
5. **Does the SWA:global ratio behave like the linear:full ratio?** Both are called "3:1" and the
   literature treats them as comparable. Run both against the same recall battery at matched
   parameter *and* matched runtime-state budget. If they diverge, a large fraction of the
   cross-paper ratio comparisons in this note are invalid.
6. **Stochastic window size:** 2509.24552 reports it beats fixed windows on both short and long
   context. It is a training-time change with no inference cost — the cheapest listed idea to
   test and the easiest to fold into Proteus's config surface.
7. **Head-wise vs layer-wise at matched full-attention FLOPs.** HydraHead's 7:1-head ≈ 3:1-layer
   claim is one unreplicated paper. At 300M this is a single controlled comparison.
8. **Does the asymptotic-4× KV formula hold empirically?** Instrument actual peak cache residency
   under our allocator against `L/L_global` and find where paging, fragmentation, and the
   sub-32 GiB chunk constraint break the analytic prediction. This is a rig-validation test as
   much as a research one.

---

## Sources

**Read from our own artifacts (`[M]`)**
- Laguna S 2.1 shipped config at `b0a9fd7c850e` — 48 layers, 12 full + 36 sliding, G-S-S-S, `sliding_window` 512, 8 KV heads, `head_dim` 128. `ASSUMPTIONS.md → reference-model`, `research/reference/PROVENANCE.md`.
- `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:343, 365, 366` — per-layer head counts; layer-type list lookup; window assignment. `CODE_MAP.md`.
- `architecture/llama-cpp-laguna/src/models/laguna.cpp:41, 184`; `src/llama-kv-cache-iswa.cpp:73` — `set_swa_pattern(4, dense_first)`; per-layer RoPE divergence; two-tier cache sizing. `CODE_MAP.md`.
- `architecture/samba/lit_gpt/model.py:321, 323, 452`; `lit_gpt/config.py:33, 409` — modulo layer typing; `mb_per_layer=2`; `full_per_layer` default; explicit-position escape hatch. `CODE_MAP.md`.
- Zamba2 `mamba_block.py:445-453` — uniform Mamba stack with 2 weight-tied shared attention blocks invoked on a 54-entry schedule. `CODE_MAP.md`.
- `notebook/uma-carveout-controls-fast-tier.md` — ~200 GB/s flat to ≥62 GiB, single run per arm, 2026-07-26.
- `ASSUMPTIONS.md` rows: `reference-model`, `kv-per-token-laguna`, `laguna-heads-uniform`, `gpu-fast-tier-size`, `large-tensor-fault-32gib`, `bf16-numerics-unproven`, `ablation-scale-sufficient`.

**arXiv (`[C]`) — all ids resolved against the live arXiv API on 2026-07-26**
- 2305.13245 — GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (2023)
- 2312.00752 — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (2023)
- 2312.04927 — Zoology: Measuring and Improving Recall in Efficient Language Models (2023)
- 2306.14048 — H2O: Heavy-Hitter Oracle for Efficient Generative Inference (2023)
- 2402.18668 — Simple linear attention language models balance the recall-throughput tradeoff (2024)
- 2403.17844 — Mechanistic Design and Scaling of Hybrid Architectures (2024-03-26)
- 2403.19887 — Jamba: A Hybrid Transformer-Mamba Language Model (2024-03)
- 2404.06654 — RULER: What's the Real Context Size of Your Long-Context Language Models? (2024)
- 2404.14469 — SnapKV: LLM Knows What You are Looking for Before Generation (2024)
- 2405.16712 — Zamba: A Compact 7B SSM Hybrid Model (2024-05-26)
- 2405.21060 — Transformers are SSMs: Generalized Models and Efficient Algorithms Through SSD (2024)
- 2406.07522 — Samba: Simple Hybrid State Space Models for Efficient Unlimited Context (2024-06-11)
- 2406.07887 — An Empirical Study of Mamba-based Language Models (Waleffe et al., 2024-06-12)
- 2411.13676 — Hymba: A Hybrid-head Architecture for Small Language Models (2024-11-20)
- 2411.15242 — The Zamba2 Suite: Technical Report (2024-11-22)
- 2412.06464 — Gated Delta Networks: Improving Mamba2 with Delta Rule (2024-12-09)
- 2503.19786 — Gemma 3 Technical Report (2025-03-25)
- 2504.03624 — Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models (2025-04)
- 2504.17768 — The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs (2025-04-24)
- 2504.21463 — RWKV-X: A Linear Complexity Hybrid Language Model (2025-04-30)
- 2507.06457 — A Systematic Analysis of Hybrid Linear Attention (2025-07-08, rev. 2026-06-24)
- 2508.10925 — gpt-oss-120b & gpt-oss-20b Model Card (2025-08-08)
- 2508.15884 — Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search (2025-08-21)
- 2509.24552 — Short window attention enables long-term memorization (2025-09-29, rev. 2026-05-04)
- 2510.04800 — Hybrid Architectures for Language Models: Systematic Analysis and Design Insights (2025-10-06, rev. 2026-04-21)
- 2510.05901 — Untangling Component Imbalance in Hybrid Linear Attention Conversion Methods (2025-10-07)
- 2510.19338 — Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning (Ring-linear, 2025-10-22)
- 2510.26692 — Kimi Linear: An Expressive, Efficient Attention Architecture (2025-10-30)
- 2601.02780 — MiMo-V2-Flash Technical Report (2026-01-06)
- 2601.22156 — Hybrid Linear Attention Done Right: Efficient Distillation and Effective Architectures for Extremely Long Contexts (HALO/HypeNet, 2026-01-29)
- 2602.03560 — HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection and KV Cache Sharing (2026-02-03)
- 2603.15569 — Mamba-3: Improved Sequence Modeling using State Space Principles (2026-03-16)
- 2604.03444 — Olmo Hybrid: From Theory to Practice and Back (2026-04-03)
- 2605.09516 — Mixture of Layers with Hybrid Attention (2026-05-10)
- 2605.20936 — DASH: Fast Differentiable Architecture Search for Hybrid Attention in Minutes on a Single GPU (2026-05-20)
- 2605.26494 — The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence (2026-05-26)
- 2606.13392 — MiniMax Sparse Attention (2026-06-11)
- 2606.15378 — Rethinking the Role of Efficient Attention in Hybrid Architectures (2026-06-13)
- 2606.18056 — ConSA: Controllable Sparsity in Hybrid Attention via Learnable Allocation (2026-06-16)
- 2606.20097 — HydraHead: From Head-Level Functional Heterogeneity to Specialized Attention Hybridization (2026-06-18)
- 2606.30562 — Morphing into Hybrid Attention Models (FlashMorph, 2026-06-29)
- 2607.07953 — Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing (2026-07-08)

**Non-arXiv (`[C]`)**
- MiniMax, "Why Did MiniMax M2 End Up as a Full Attention Model?", 2025-10-30 — https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model (vendor retrospective, not peer-reviewed)
- M. Labonne, "Qwen3.5: Nobody Agrees on Attention Anymore", Feb 2026 — https://huggingface.co/blog/mlabonne/qwen35 (secondary; used only for the Feb-2026 release dates and shipped ratios of Qwen3.5, Kimi K2.5, GLM-5, MiniMax M2.5)
- Qwen3-Next-80B-A3B model card — https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct (3:1 Gated DeltaNet : Gated Attention; no ratio ablation published)

**Not verified against a primary source — do not quote without checking**
- Kimi Linear ratio-ablation perplexity values (5.65 / 5.66 / 5.70 / 5.77 / 5.82 for 3:1 / 1:1 / 7:1 / 0:1 / 15:1). Reported by secondary summaries of 2510.26692; the arXiv HTML render fails and the PDF was not extractable in this environment.
- 2510.04800's per-axis numeric conclusions on inter- vs intra-layer fusion. Only the abstract was read this session.
