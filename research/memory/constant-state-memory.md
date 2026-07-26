# Constant-state memory — SSMs and linear attention

This note settles what "constant state" actually means in bytes and in bandwidth, and
establishes that every architecture in this family — Mamba-2/3, DeltaNet and Gated
DeltaNet, RWKV-7, mLSTM/xLSTM, Lightning Attention — is the *same* recurrence differing
only in what it is allowed to do to the state before writing. It settles that the
recall failure is real, reproducible, and measurable, with numbers: on the RULER
word-in-haystack needle at 1.3B params, Mamba-2 scores 64.4 at 1K context and **4.6 at
4K** `[C]`, while a 70M-parameter attention model beats a 1.4B gated-convolution model
on associative recall `[C]`. It does **not** settle whether constant state is the right
target at all — three separate 2026 lines of work abandon strict O(1) on purpose, and
two well-resourced labs published opposite production conclusions in the same year.

---

## 1. One recurrence, six architectures

Write `t` for token index, `T` for sequence length. For one head:

- `k_t ∈ R^{d_k}` — the **key**: the address this token files its content under.
- `v_t ∈ R^{d_v}` — the **value**: the content being filed.
- `q_t ∈ R^{d_k}` — the **query**: the address being looked up at this step.
- `S_t ∈ R^{d_v × d_k}` — the **state**. A matrix. This is the entire memory of
  everything before token `t`. Its shape has no `T` in it. That is the whole trick.

The write and the read:

```
S_t = M_t · S_{t-1}  +  β_t · v_t k_tᵀ        (write)
o_t = S_t · q_t                                (read)
```

`v_t k_tᵀ` is an **outer product** — the d_v×d_k matrix whose (i,j) entry is
`v_t[i]·k_t[j]`. It has rank 1: it is the cheapest possible matrix that means "content
`v` at address `k`". `β_t` is write strength. `M_t` is a transition operator applied to
the *existing* memory before the new write lands — the forget/edit step.

Reading is a matrix–vector product, and this is where the family lives or dies. Suppose
`M_t = I` and `β_t = 1`, so `S_T = Σ_j v_j k_jᵀ`. Then

```
o = S_T q = Σ_j v_j (k_jᵀ q)
```

You get **every value ever written**, each scaled by the dot product between its key and
your query. If the keys were mutually orthogonal unit vectors and `q = k_5`, every term
vanishes except `v_5` and the read is exact. They are not orthogonal — they are learned
continuous vectors in `d_k` dimensions, and you cannot pack more than `d_k` mutually
orthogonal directions into `R^{d_k}`. Past that point every read returns a weighted
blend of the thing you wanted and everything else. **That is the failure mode, and it is
not a miss — it is a confident wrong answer.**

Softmax attention avoids this by not superposing at all: it keeps `t` separate `(k,v)`
pairs (the KV cache) and applies `softmax` over the scores, a sharpening nonlinearity
that suppresses the non-matching terms. Exact log plus sharp read, at O(T) bytes and
O(T) reads per token. The trade is stated exactly there.

**What each architecture puts in `M_t`:** (the unified-notation comparison of the
linear half — DeltaNet, Gated DeltaNet, KDA, Gated DeltaNet-2 — on expressivity, decay,
and throughput is `[C]` 2607.07953. The move that made any of this competitive was
Mamba's input-dependent *selective* gating plus a hardware-aware scan `[C]` 2312.00752 —
the systems half of that paper is as load-bearing as the math half.)

| Architecture | `M_t` | Read as |
|---|---|---|
| Linear attention (Katharopoulos-style) | `I` | Accumulate forever, never forget |
| Lightning Attention `[C]` 2401.04658 | `I` (or scalar decay) | **Not a new recurrence** — an I/O-aware tiled *kernel* for the above |
| Mamba-2 / SSD `[C]` 2405.21060 | `a_t · I`, scalar `a_t = exp(−Δ_t·exp(A_log)) ∈ (0,1)` | Global TTL tick on the whole matrix |
| mLSTM (xLSTM) `[C]` 2405.04517 | `f_t · I` with **exponential** gating, plus a separate normalizer state `n_t` | Same, but the read is normalized so unbounded accumulation cannot blow up |
| DeltaNet `[C]` 2406.06484 | `I − β_t k_t k_tᵀ` (a Householder-type reflection) | Read-before-write: erase exactly this key's direction, then write |
| Gated DeltaNet `[C]` 2412.06464 | `α_t (I − β_t k_t k_tᵀ)` | Both: global decay *and* targeted overwrite |
| Gated DeltaNet-2 `[C]` 2605.22791 | channel-wise **erase** gate `b_t` and channel-wise **write** gate `w_t`, decoupled | The scalar `β` was doing two jobs; split them |
| RWKV-7 `[C]` 2503.14456 | diagonal-plus-rank-one, vector-valued decay + in-context learning rate | Per-channel TTL rather than one scalar |
| Mamba-3 `[C]` 2603.15569 | **complex-valued** scalar — decay *and rotation* | A state that can rotate can count; a state that can only decay cannot |

The delta term is worth dwelling on because it is the one genuinely new idea since 2023.
In the flash-linear-attention reference implementation the whole of Gated DeltaNet's
memory update is three lines: decay the matrix, read back what it currently returns for
this key, then write the *residual*
(`architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:54,56,58` — see
`research/reference/CODE_MAP.md`). `β=1` is an exact overwrite of that key's slot, `β=0`
a no-op. It is compare-and-swap against whatever is currently resident.

**A correction to the folk story, read out of the code rather than the paper:** the
"gated" half of Gated DeltaNet cannot forget selectively. The decay is *one scalar per
head applied to the entire K×V matrix* — every stored association is attenuated by
exactly the same factor every step. The kernel already implements per-key and
per-channel gates (`fused_recurrent.py:138-150`) and the layer simply does not pass
them. All content-dependent selectivity lives in the delta term. So "gating = selective
forgetting, delta = accumulation" is backwards: **the gate is indiscriminate decay, the
delta is the targeted erase.** Gated DeltaNet-2 and RWKV-7 are both, in effect, fixes for
this.

---

## 2. What constant state buys, in bytes and in bandwidth

Gated DeltaNet-2 states its per-layer recurrent state exactly: `H·d_k·d_v = 16·128·128 =
262,144 floats per batch element` `[C]` 2605.22791. At bf16 that is **512 KiB per layer
per sequence** — at 1K context, at 1M context, identically.

Set that against the reference model. Laguna S 2.1 costs `2·48·8·128·2 B = 192 KiB/token`
if every layer were global attention, giving **24 GiB of KV at 128k context** `[M]`
(`ASSUMPTIONS.md → kv-per-token-laguna`, computed from the fetched config; the shipped
3:1 SWA/global interleave cuts real steady-state residency by roughly 4×, so 24 GiB is
the all-global bound rather than what you actually hold). A 48-layer constant-state model
at 512 KiB/layer would hold **24 MiB**. Three orders of magnitude, and the gap widens
linearly with context.

The bandwidth consequence is the one a storage engineer feels immediately. Decode reads
the entire cache once per generated token. Our measured fast tier runs at **~200 GB/s out
to ≥62 GiB** `[M]` (`notebook/uma-carveout-controls-fast-tier.md`, single run per arm).
Arithmetic on those measured inputs: 24 GiB ≈ 25.8 GB of KV traffic per token → 0.129 s
→ a **hard ceiling near 8 tokens/s from cache traffic alone**, before a single FLOP of
useful work. The 24 MiB constant state reads in 0.13 ms and disappears into the noise;
decode then becomes weight-bound, which is a fixed cost you can amortize with batching.
This is why the constant-state literature exists at all, and it is a capacity-planning
argument, not a modeling one.

State size is a dial, not a property of the architecture family, and Based `[C]`
2402.18668 is the paper that says so most directly: vary the sliding-window size and the
linear-attention feature dimension and you traverse the recall-versus-state-size Pareto
frontier, recovering full attention quality at one end and a small fixed state at the
other. At 1.3B it beats the strongest sub-quadratic baselines on recall-intensive tasks
by **6.22 accuracy points** and reaches **24× the generation throughput of
FlashAttention-2** at 1024 generated tokens. Read it as capacity planning: state is a
budget you spend, and someone has already drawn the curve.

The 2026 frontier has stopped treating this as binary. Sparse Delta Memory raises the
state **~3 orders of magnitude at identical FLOPs** — at 8B scale roughly 8.0B state
parameters against Gated DeltaNet's 2.2M, using 64 sparse reads and 64 sparse writes per
step instead of a dense outer product `[C]` 2607.07386. State size and per-token compute
turn out to be separable knobs. That is a memory hierarchy, spelled out.

---

## 3. The recall failure, concretely

### MQAR — the diagnostic

Multi-query associative recall, from Zoology `[C]` 2312.04927: present key–value pairs,
then query several of them, out of order. `A 4 B 3 C 6 F 1 E 2 → A ? C ? F ? E ? B ?`
must produce `4, 6, 1, 2, 3`. It is deliberately *multi*-query because single-query
induction-head tasks were solved by architectures that still failed on real text.

Zoology's numbers, from 17 pretrained models:

- State-of-the-art gated-convolution architectures underperform attention by **up to 2.1
  perplexity points on the Pile**, and **82% of that gap** is on tokens requiring
  associative recall.
- On the AR slice, a **70M-parameter Transformer reaches 2.41 ppl against a 1.4B Hyena's
  3.43 ppl** — a 20× larger model, decisively worse.
- Attention solves MQAR at a constant model dimension of **64** across all tested
  sequence lengths. Gated convolutions "do not achieve accuracy >0.9 unless **d ≥ N**" —
  model dimension must scale with sequence length. That is the capacity wall in one
  inequality.
- Hybrids with input-dependent sparse attention close **97.4%** of the gap.

### RULER needles — the failure at production scale

The cleanest concrete numbers come from the Gated DeltaNet paper's Table 3, all models
**1.3B params, 100B FineWeb-Edu tokens, 4K training length** `[C]` 2412.06464:

**S-NIAH-2 (number in haystack)**, accuracy at 1K / 2K / 4K / 8K:

| Model | 1K | 2K | 4K | 8K |
|---|---|---|---|---|
| DeltaNet | 98.4 | 45.6 | 18.6 | 14.4 |
| Mamba-2 | 99.4 | 98.8 | 56.2 | 17.0 |
| Gated DeltaNet | 100.0 | 99.8 | 92.2 | 29.6 |

**S-NIAH-3 (word in haystack — higher-entropy value)**, at 1K / 2K / 4K:

| Model | 1K | 2K | 4K |
|---|---|---|---|
| DeltaNet | 85.2 | 47.0 | 22.4 |
| Mamba-2 | 64.4 | 47.6 | **4.6** |
| Gated DeltaNet | 86.6 | 84.2 | 27.6 |

Read those two tables together and the mechanism is legible. Mamba-2's pure scalar decay
holds a low-entropy needle well (98.8 at 2K on S-NIAH-2) and collapses on a
high-entropy one (4.6 at 4K on S-NIAH-3) — it has capacity but no way to clear a slot.
DeltaNet has targeted overwrite but no decay and saturates early. Gated DeltaNet has both
and dominates until 8K, where it also falls to 29.6. **Note the training length is 4K**:
the 4K column is not extrapolation. This is capacity failing inside the trained regime.

Multi-key needles are harder still. From Gated DeltaNet-2's Table 3, same 1.3B/100B
setup, MK-NIAH-1 recurrent at 1K / 2K / 4K `[C]` 2605.22791:

| Model | 1K | 2K | 4K |
|---|---|---|---|
| Mamba-2 | 59.2 | 38.6 | 14.4 |
| Gated DeltaNet | 89.8 | 54.2 | 60.6 |
| KDA | 77.4 | 63.2 | 26.2 |
| Mamba-3 (SISO) | 60.2 | 35.6 | 12.2 |
| Mamba-3 (MIMO) | 89.2 | 72.4 | 29.2 |
| Gated DeltaNet-2 | 92.0 | 89.8 | 31.8 |

Every architecture in the family — including the two published in 2026 — is under 32% at
4K on multi-key retrieval. Also worth noting: Mamba-3 (SISO) is the *worst* row here
despite being the newest and beating Gated DeltaNet by 0.6pp on average downstream
accuracy at 1.5B `[C]` 2603.15569. Aggregate downstream accuracy and retrieval capacity
are not the same measurement and do not rank models the same way.

### The theory: why this cannot be engineered away

Three independent 2026 statements of the same bound, at different levels of rigour:

- **Information-theoretic.** "The Impossibility Triangle of Long-Context Modeling"
  `[C]` 2605.05066 (May 2026) proves via the Data Processing Inequality and Fano's
  Inequality that no model achieves all three of: per-step compute independent of
  sequence length, state size independent of sequence length, and recall of a number of
  facts proportional to sequence length. A model with the first two recalls at most
  `O(poly(d)/log V)` key–value pairs. *Caveat: single-author preprint, unreplicated.*
  For a distributed-systems reader this is a CAP-shaped result and should be treated with
  the same care — the theorem is only as useful as the fidelity of its model.
- **Coding-theoretic.** KATA `[C]` 2607.17419 (Jul 2026) recasts associative recall as
  spherical packing and characterizes capacity by the **Welch interference floor** — the
  minimum achievable mutual coherence of `n` unit vectors in `d` dimensions. Above that
  floor you can pack exponentially many keys; below it, interference is mandatory.
- **Empirical.** Zoology's `d ≥ N` requirement, above.

### A cautionary MQAR table

Variational Linear Attention `[C]` 2605.11196 (May 2026) reports an MQAR capacity curve
at head dimension `d_h = 32` (Table 11, seed 42, 1000 training steps):

| n pairs | VLA | DeltaNet | Linear | Softmax |
|---|---|---|---|---|
| 8 | 0.997 | 0.965 | 0.150 | 0.152 |
| 16 | 0.990 | 0.009 | 0.091 | 0.091 |
| 24 | 0.994 | 0.007 | 0.069 | 0.070 |
| 32 (= d_h) | 0.623 | 0.008 | 0.056 | 0.057 |
| 48 | 0.044 | 0.008 | 0.043 | 0.043 |

The shape — perfect below `n < d_h`, cliff at the capacity boundary — is exactly what the
theory predicts. **But the softmax column scores 0.152 at n=8**, on a task Zoology shows
attention solving essentially perfectly at model dimension 64. A broken control means the
DeltaNet column cannot be read as a property of DeltaNet either. Reproduced here as a
worked example of the house rule: *if a result looks wrong, suspect the harness first.*
Two-author preprint, 1000 training steps, one seed.

More generally: **none of the tables in this section report seed counts or confidence
intervals** in the material I read. By this lab's own standard they are anecdotes with
good provenance, not measurements.

---

## 4. Where the systems analogy breaks

The bridge is obvious and mostly correct: a fixed-size state is a fixed-size cache, decay
is a TTL, `β` is write strength, the delta term is a compare-and-swap. Four places it
breaks, and the breaks are the part that teaches.

**1. There is no miss signal.** A cache tells you it missed; that is the single most
useful thing it does, because a miss is a *typed event* you can count, alarm on, and
service. A recurrent state returns a plausible blend of everything it holds and cannot
distinguish "I have this" from "I have three things that rhyme with this." Hit rate is
not merely unmeasured, it is undefined. Every observability instinct you own assumes a
distinguishable miss.

**2. There is no backing store, therefore no promotion, demotion, or fault path.** The
decay multiply *destroys* content rather than relocating it
(`architecture/mamba/mamba_ssm/ops/triton/ssd_state_passing.py:80` is the entire
inter-chunk recurrence — a decay and an add, in place). Token 5's contribution cannot be
recovered at token 5000 because it was not evicted to a slower tier, it was overwritten.
An LLM KV cache is an append-only exact log you can re-scan; this is unreplayable.

**3. There are no addresses and no lines.** Keys are L2-normalized continuous vectors, so
every write smears across the whole matrix and a *similar* key partially clobbers a
neighbour's content. This is a fully-associative store with lossy superposition, not a
set-associative cache. There is no capacity miss because there is no capacity check.

**4. "Constant memory" is a decode-time property only.** During training and prefill the
chunked scan materializes one state per chunk before the boundary pass runs, so activation
memory is `O(L / chunk_size)`, not `O(1)`. And that per-chunk state is computed in fp32
regardless of model dtype (`ssd_combined.py:375`, `states_in_fp32=True`) — the
constant-size state is quietly the most numerically fragile part of the layer. On
gfx1151, where `bf16-numerics-unproven` is still open `[M]`/untested, that is a live risk,
not a footnote.

A fifth, smaller one: **Lightning Attention is not an architecture.** It is an I/O-aware
tiled kernel — conventional attention for intra-block terms, the linear-attention kernel
trick for inter-block terms `[C]` 2401.04658 — standing in the same relation to linear
attention that FlashAttention stands to attention. Listing it beside Mamba-2 as a
"variant" conflates a performance change with a capability change. MiniMax-01 shipped it
at 456B total / 45.9B active params with 1M training context and 4M inference context
`[C]` 2501.08313, in a reported 7 lightning : 1 softmax per 8-layer block (that ratio is
from MiniMax's release materials and third-party analyses, not read out of the paper text
in this session).

---

## 5. The four escape hatches, all opened in the last 18 months

1. **Hybrids — keep some full attention.** The industry default converged on **3:1
   linear-to-full**: Kimi Linear `[C]` 2510.26692 (3:1 KDA:MLA, 75% KV reduction, 6× decode
   throughput at 1M context), Qwen3-Next (every 4th layer full attention). The systematic
   study `[C]` 2507.06457 trained 72 models (36 @340M/20B tokens, 36 @1.3B/100B) across 6
   linear variants × 5 ratios and recommends 3:1–6:1, finding recall improves sharply as
   you go *below* 3:1. Our own reference model is measured at 3:1 — 12 full + 36 sliding
   in a strict GSSS pattern `[M]` (`ASSUMPTIONS.md → reference-model`).
2. **Grow the state slowly instead of not at all.** Log-Linear Attention `[C]` 2506.04761
   (ICLR 2026) replaces the fixed state with a logarithmically growing set of states,
   log-linear compute in `T`. This directly abandons the O(1) premise of this whole note.
3. **Make the state huge but sparsely addressed.** Sparse Delta Memory `[C]` 2607.07386:
   at 1.4B, RULER average 31.2 (SDM) vs 20.0 (Gated DeltaNet) vs 32.5 (full attention); at
   8B, 50.2 vs 34.2 vs 61.2. Note the pattern — sparse addressing nearly closes the gap at
   1.4B and reopens it at 8B. Sparse State Expansion `[C]` 2507.16577 attacks the same axis
   by partitioning the state.
4. **Keep a small exact cache alongside the lossy state.** HOLA `[C]` 2607.02303 (Jul 2026)
   pairs a delta-rule compressive state with a *bounded exact KV cache*, admitting tokens
   by residual magnitude `β·||e||`. At 340M / 15B SlimPajama tokens: Wikitext ppl 27.32 →
   22.92 (−16.1%), below a full-attention Transformer++ at 26.88, with RULER needle recall
   holding to 32k (16× training length). Single-author, unreplicated — but this is the
   paper closest in shape to a Mnemosyne contribution, because it is explicitly a
   two-tier memory with an admission policy.

Note what hatches 2–4 have in common: **all three concede that a strictly constant state
is the wrong constraint**, and re-introduce a growing or explicitly-addressed tier. The
research question has quietly moved from "can we do without a KV cache" to "what is the
right size and admission policy for the exact tier."

---

## 6. Contested — do not let this note imply consensus

- **Does hybrid linear attention work at production scale?** MiniMax shipped M2 (229.9B
  total / 9.8B active `[C]` 2605.26494) as **full attention with plain MHA**, publishing a
  note stating they found no efficient-attention variant that reliably matched full
  attention across reasoning, coding and agentic tasks; hybrids matched on MMLU/BBH/MATH/
  LongBench but showed clear deficits on multi-hop reasoning at scale, across hundreds of
  billions to trillions of continued-pretraining tokens `[C]` (minimax.io, "Why Did M2 End
  Up as a Full Attention Model?"; third-party analysis at lmsys.org, Nov 2025). Kimi Linear
  `[C]` 2510.26692 claims the opposite under matched-scale pretraining. **Both are
  shipping-product retrospectives with commercial incentives; neither is a controlled
  ablation.** Treat the disagreement as the open question of the track.
- **Does the hybrid ratio set a capability ceiling or only a training-speed knob?**
  2507.06457 finds recall improves sharply as full-attention layers increase, *particularly
  below* a 3:1 ratio — quoted from the abstract, 2026-07-26, after an earlier draft of this
  note stated the direction backwards in this bullet; 2606.15378 `[C]` argues
  different configurations converge given enough training and that the efficient-attention
  choice governs *how fast* long-context ability emerges, not its limit. Same year,
  incompatible framings. The resolution likely depends on token budget — which is exactly
  the axis a small-scale rig can attack.
- **Is state-tracking expressivity the same thing as recall?** RWKV-7 sells recognizing all
  regular languages, beyond the TC⁰ limit of Transformers `[C]` 2503.14456; Mamba-3 sells
  complex-valued state for state tracking `[C]` 2603.15569; the underlying theory is that
  eigenvalues in `[−1,1]` rather than `[0,1]` unlock parity and, for products of
  `I − vvᵀ` matrices, any regular language `[C]` 2411.12537. **None of that is associative
  recall capacity**, which is bounded by state size regardless of eigenvalue range. The
  literature routinely presents both under "expressivity" and they are different axes.
  Mamba-3 (SISO) scoring 12.2 on 4K multi-key retrieval while winning on downstream average
  is the empirical form of that distinction.
- **Do MQAR results transfer?** Zoology attributes 82% of a real perplexity gap to AR, which
  is the strongest transfer evidence available. But MQAR is synthetic, and no source I read
  demonstrates that an MQAR capacity curve predicts a downstream task curve quantitatively.
- **Matched params or matched state?** Most comparisons match parameter count; Gated
  DeltaNet-2 argues matched *state size* is the correct control `[C]` 2605.22791. These give
  different rankings, and papers rarely report both.

---

## Open questions

Testable at 20M–300M params on one gfx1151 GPU, ≥62 GiB fast tier at ~200 GB/s `[M]`,
single-device only, individual tensors kept under 32 GiB `[M]`.

1. **Does the `n_pairs < d_h` capacity cliff reproduce on our silicon, and is it sharp?**
   MQAR is synthetic and tiny; this is a one-day experiment and the natural first
   calibration of the rig against a published shape.
2. **Holding parameter count fixed, does state size alone move the S-NIAH cliff?** Sweep
   `d_k × d_v` in a Gated DeltaNet at 20M–300M. This is the matched-state-vs-matched-params
   dispute, run as an ablation rather than argued.
3. **Attribution: is the cliff caused by decay or by interference?** Ablate the gate
   (`α = 1`) and the delta term (`β` fixed vs learned) *separately* and see which one moves
   the cliff. Directly tests the code-derived claim that the gate is indiscriminate and the
   delta is the targeted erase.
4. **Can the missing miss signal be synthesized?** Train a cheap probe on the state (or on
   `||S q||` / read-margin statistics) to predict "this read is unreliable." If a usable
   confidence signal exists, everything a systems engineer knows about fallback paths
   becomes applicable — and this is the most Mnemosyne-shaped question in the note.
5. **Does 3:1 hold at 20M–300M?** 2507.06457's smallest arm is 340M, just above our
   ceiling. If the optimal ratio drifts with scale, every small-scale hybrid ablation in the
   literature is mis-calibrated, including ours.
6. **What is the smallest exact cache that recovers full-attention MQAR?** Sweep HOLA-style
   bounded-cache size against recall at fixed state size. The output is a capacity-planning
   curve denominated in bytes — directly usable against the 62 GiB budget.
7. **Does a bf16 recurrent state change the recall cliff?** Mamba's chunk states are fp32 by
   construction. Given `bf16-numerics-unproven` is still open on this hardware, run the
   state in bf16 vs fp32 and measure recall, not just loss. Ties the Hardware Validation
   Gate to a research result instead of treating it as maintenance.
8. **Does the prefill state materialization hit our 32 GiB single-tensor fault?** Chunked
   scan holds `O(L / chunk_size)` fp32 states. At long `L` and small `chunk_size` this is a
   real tensor with a known hard failure mode on this machine — a hang at 0 CPU with no
   error `[M]`.

---

## Sources

**Verified against the arXiv API in `research/reference/papers/resolved_papers.json`:**

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces — arXiv 2312.00752 (Dec 2023)
- Zoology: Measuring and Improving Recall in Efficient Language Models — arXiv 2312.04927 (Dec 2023)
- Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2 / SSD) — arXiv 2405.21060 (May 2024)
- Simple linear attention language models balance the recall-throughput tradeoff (Based) — arXiv 2402.18668 (Feb 2024, rev. Mar 2025)
- Parallelizing Linear Transformers with the Delta Rule over Sequence Length (DeltaNet) — arXiv 2406.06484 (Jun 2024)
- Gated Delta Networks: Improving Mamba2 with Delta Rule — arXiv 2412.06464 (Dec 2024)
- Mamba-3: Improved Sequence Modeling using State Space Principles — arXiv 2603.15569 (Mar 2026)
- Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity — arXiv 2607.07386 (Jul 2026)
- A Systematic Analysis of Hybrid Linear Attention — arXiv 2507.06457 (Jul 2025, rev. Jun 2026)
- Kimi Linear: An Expressive, Efficient Attention Architecture — arXiv 2510.26692 (Oct 2025)
- Rethinking the Role of Efficient Attention in Hybrid Architectures — arXiv 2606.15378 (Jun 2026)
- Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing — arXiv 2607.07953 (Jul 2026)

**Abstract pages fetched directly during this session:**

- RWKV-7 "Goose" with Expressive Dynamic State Evolution — arXiv 2503.14456 (Mar 2025)
- xLSTM: Extended Long Short-Term Memory — arXiv 2405.04517 (May 2024)
- Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models — arXiv 2401.04658 (Jan 2024)
- MiniMax-01: Scaling Foundation Models with Lightning Attention — arXiv 2501.08313 (Jan 2025)
- Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention — arXiv 2605.22791 (May 2026)
- Log-Linear Attention — arXiv 2506.04761 (Jun 2025, ICLR 2026, rev. Mar 2026)
- Kernelized Linear Attention: Breaking the Capacity Wall with Symmetric Cones (KATA) — arXiv 2607.17419 (Jul 2026)
- A Hippocampus for Linear Attention: An Exact Memory for What the Recurrent State Forgets (HOLA) — arXiv 2607.02303 (Jul 2026) — *single author, unreplicated*
- Variational Linear Attention: Stable Associative Memory for Long-Context Transformers — arXiv 2605.11196 (May 2026) — *two authors, one seed, softmax control appears broken*
- The Impossibility Triangle of Long-Context Modeling — arXiv 2605.05066 (May 2026) — *single author, unreplicated*
- Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues — arXiv 2411.12537 (Nov 2024, rev. Mar 2025)
- Scaling Linear Attention with Sparse State Expansion — arXiv 2507.16577 (Jul 2025, rev. Oct 2025)
- The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence — arXiv 2605.26494 (May 2026)

**Non-arXiv:**

- MiniMax, "Why Did M2 End Up as a Full Attention Model?" — https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model
- LMSYS, "No Free Lunch: Deconstruct Efficient Attention with MiniMax M2" — https://www.lmsys.org/blog/2025-11-04-miminmax-m2/ (Nov 2025)
- Alibaba Cloud, "Qwen3-Next: Towards Ultimate Training & Inference Efficiency" — https://www.alibabacloud.com/blog/602580 (3:1 Gated DeltaNet to full attention; every 4th layer full)
- MiniMax-01's reported 7:1 lightning-to-softmax layer ratio comes from MiniMax release
  materials and third-party analyses, not from the paper text read in this session.

**In-repo:**

- `research/reference/CODE_MAP.md` — "Mamba-2 SSD: the chunked scan and the fixed-size
  carry" and "Gated DeltaNet: the state update rule (decay tick + delta write)". All
  `file:line` pointers used above are machine-verified against the pinned revisions in
  `research/reference/PROVENANCE.md`.
- `ASSUMPTIONS.md` — `gpu-fast-tier-size` (≥62 GiB at ~200 GB/s), `kv-per-token-laguna`
  (192 KiB/token exact; 24 GiB at 128k is the all-global bound), `large-tensor-fault-32gib`,
  `bf16-numerics-unproven`, `reference-model` (3:1 GSSS interleave), `single-device-only`.
- `notebook/uma-carveout-controls-fast-tier.md` — the bandwidth measurement.
