# Long-context behaviour: RoPE, YaRN, NoPE, and why 1M advertised is not 1M useful

This note settles three things. **First**, the positional machinery: what RoPE actually
computes, why it fails past the training window, and exactly what Position Interpolation,
NTK-aware scaling and YaRN change — written out in symbols, with Laguna's own numbers
computed from its committed config rather than quoted from a paper. **Second**, the
architecture-level answer: Laguna ships `max_position_embeddings: 1048576` with a YaRN
`factor` of 128 applied to `full_attention` layers **only**, and it rotates just half of
each global head's channels, which makes it simultaneously a per-layer RoPE-flavour hybrid
and a per-channel NoPE hybrid — a design the length-generalization literature predicts and
that the model card compresses into four words. **Third**, the measurement discipline:
advertised context is an arithmetic consequence of a config field, effective context is a
measurement against a threshold, and every serious benchmark from RULER (2024) to ATLAS
(2026) finds the second well below the first.

---

## 1. RoPE, written out

A head with head dimension `d` splits its query vector `q ∈ R^d` into `d/2` channel pairs.
Pair `i ∈ {0 … d/2−1}` is assigned an **angular frequency**

```
ω_i = θ^(−2i/d)          radians per token
```

`θ` is the *rope base* (Laguna global layers: `θ = 500000`; sliding layers: `θ = 10000`).
At absolute position `m` the pair is rotated by angle `m·ω_i`:

```
[q'_2i  ]   [ cos(m ω_i)  −sin(m ω_i) ] [q_2i  ]
[q'_2i+1] = [ sin(m ω_i)   cos(m ω_i) ] [q_2i+1]
```

Keys get the same treatment at their own position `n`. Because a rotation matrix is
orthogonal, `⟨R_m q, R_n k⟩ = ⟨q, R_{n−m} k⟩` — the logit depends only on the **relative**
distance `n−m`, even though the code only ever handles absolute positions.
`[C]` RoFormer, arXiv 2104.09864 (2021). (HuggingFace splits the vector into halves rather
than adjacent pairs — `rotate_half` — which is the same computation under a channel
permutation.)

The unit that matters is not frequency but **wavelength**:

```
λ_i = 2π / ω_i = 2π · θ^(2i/d)     tokens per full rotation
```

**Systems bridge.** This is a mixed-radix odometer, or a bank of clocks with geometrically
spaced periods. Position is not stored as an integer; it is *encoded as phase* across
`d/2` dials running at very different speeds. Fast dials distinguish "is this the previous
token or the one before it"; slow dials distinguish "is this in the same paragraph or the
same file". Reading a relative distance means reading the whole dial bank at once —
positional information is distributed, redundant and error-tolerant in exactly the way a
Bloom filter or an erasure code is.

**Where the analogy breaks, and it is the load-bearing break for Mnemosyne.** An odometer
can be re-read at any time. A KV cache stores **post-rotation** keys — `apply_rotary_pos_emb`
runs before the cache write — so each cached key has its absolute position baked in as
phase. This is the equivalent of writing a *physical* address into a cache line rather than
a virtual one: you cannot remap, re-pack, re-order or renumber cached entries without
recomputing them. Any eviction or compaction scheme that changes a token's effective
position is silently corrupting data, and nothing in the stack raises an error — the model
just answers wrong. Verified in code at
`research/reference/models/laguna-s/modeling_laguna.py:295-305`.

## 2. Why extension is hard: the slow dials were never trained

Laguna's global layers were pretrained at `original_max_position_embeddings: 8192`
`[M] config.json:47`. With `partial_rotary_factor: 0.5`, only `d_rot = 64` of the 128
channels are rotated, giving 32 dials. Their wavelengths, computed from the config:

| pair `i` | wavelength `λ_i` (tokens) | full rotations inside the 8192 training window |
|---|---|---|
| 0 | 6.3 | 1303.8 |
| 8 | 167 | 49.0 |
| 9 | 252 | 32.5 |
| 17 | 6 695 | 1.22 |
| 18 | 10 089 | 0.81 |
| 31 | 2 084 765 | 0.004 |

`[M]` computed 2026-07-26 from `config.json` + the `_compute_yarn_parameters` source in
`research/reference/architecture/transformers/src/transformers/modeling_rope_utils.py:423-459`.

The fast dials swept their entire phase space thousands of times during pretraining. The
slowest dial swept 0.4% of one rotation — **ever**. Asking the model for position 1,000,000
does not merely present a longer input; it presents *phase values on the slow dials that
were never observed at training time*. That is out-of-distribution input, not out-of-range
input, and it is why extending context is not a config change.

## 3. The three scalings, in order

**Position Interpolation (PI).** `[C]` arXiv 2306.15595 (Jun 2023). Replace `m` with `m/s`
for a scale factor `s`; equivalently `ω_i → ω_i/s` on every dial. Positions `0 … s·L_train`
now land inside the observed phase range. Reported to reach 32768 tokens on LLaMA with
fine-tuning inside 1000 steps. The cost is uniform: the *fast* dials, which the model uses
to tell adjacent tokens apart, also slow by `s`, so at `s = 128` neighbouring tokens differ
by 1/128 of the rotation they used to. Local discrimination collapses.

**NTK-aware scaling.** Instead of dividing every `ω`, change the base:
`θ → θ · s^(d_rot/(d_rot−2))`. Because `ω_i = θ^(−2i/d)`, a base change scales dial `i` by
roughly `s^(2i/d_rot)` — near-identity on the fast dials, near `1/s` on the slow ones. The
right shape, arrived at implicitly, and inexact at both ends.

**YaRN.** `[C]` arXiv 2309.00071 (2023). Make the split explicit and per-band, and add a
temperature. Two knobs:

*(a) Per-band ramp.* For dial `i`, let `r_i = L_orig / λ_i` be the rotations completed in
the original window. Then

- `r_i ≥ β_fast` → leave the dial completely alone (pure extrapolation),
- `r_i ≤ β_slow` → interpolate fully, `ω_i → ω_i / s`,
- in between → linear blend.

Implemented by inverting `r` to a dimension index:
`dim(r) = d_rot · ln(L_orig / (2π r)) / (2 ln θ)`
(`find_correction_dim`, `modeling_rope_utils.py:423`), then a clamped linear ramp between
the two indices (`:436`, `:451`, `:454`).

*(b) Attention temperature.* Scale the pre-softmax logits by `t = 0.1·ln(s) + 1` (applied in
practice by scaling the cached `cos`/`sin`, hence both `q` and `k`, hence the logit by `t²`).
This counteracts the entropy growth that comes from softmaxing over more keys — a global
sharpening, applied blind.

**Systems bridge.** YaRN is tiered policy applied to an addressing scheme: leave the hot
short-range index geometry untouched, re-address only the cold far tier, blend across the
boundary so there is no cliff. **Where it breaks:** a storage tiering policy can be wrong
and you find out from a hit-rate counter. Here the failure is silent and the boundary
constants — `β_fast`, `β_slow` — are inherited defaults nobody re-derives per model.

## 4. What Laguna actually ships (read from the artifact)

All of this is `[M]`, read 2026-07-26 from the committed config at
`research/reference/models/laguna-s/config.json` (revision pinned in `PROVENANCE.md`).

```json
"max_position_embeddings": 1048576,
"sliding_window": 512,
"rope_parameters": {
  "full_attention":    { "rope_type": "yarn",    "rope_theta": 500000.0, "factor": 128.0,
                         "original_max_position_embeddings": 8192,
                         "beta_fast": 32.0, "beta_slow": 1.0,
                         "attention_factor": 1.4852030263919618,
                         "partial_rotary_factor": 0.5 },
  "sliding_attention": { "rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0 }
}
```

Four things fall out, and each is checkable arithmetic rather than interpretation.

**1. The advertised context is a multiplication.** `8192 × 128 = 1,048,576`, exactly. The
"1M context" on the model card is `original_max_position_embeddings × factor`. It is a
statement about how the RoPE frequencies were rescaled, not a measurement of anything.

**2. `attention_factor` is the untuned default.** `0.1·ln(128) + 1 = 1.4852030263919618` —
matching the config to all 17 digits. Laguna took YaRN's suggested mscale and did not
search it. That is a free ablation axis sitting in plain sight.

**3. The band split, computed.** With `d_rot = 64`, `θ = 500000`, `L_orig = 8192`,
`β_fast = 32`, `β_slow = 1`: `low = 9.04 → 9`, `high = 17.49 → 18`. So of the 32 dials,
**9 are untouched, 14 are fully divided by 128, and 9 are blended**. In token terms: any
dial with wavelength under 256 tokens keeps its original speed; any dial with wavelength
over 8192 tokens is fully interpolated. `[M]`, computed this session.

**4. YaRN is on 12 of 48 layers.** Sliding layers get `rope_type: "default"`, `θ = 10000`,
full rotation of all 128 channels, and no YaRN at all. Confirmed three independent ways:
the nested config above; `modeling_laguna.py:607` building a separate `swa_rotary_emb` and
`:664-671` dispatching position embeddings by `decoder_layer.attention_type`; and
`research/reference/architecture/llama-cpp-laguna/src/models/laguna.cpp:184` onward, which
zeroes `ext_factor`, `beta_fast`, `beta_slow` and forces `attn_factor = 1.0` for SWA layers.

The consequence is the design point. **The entire 1M-token reach of this model is carried by
12 layers.** The other 36 were never given positional machinery that reaches past their
512-token window, and they do not need it: the sliding mask makes anything older
architecturally unreadable, so discarding it is lossless rather than a gamble. Read that as a
two-tier hierarchy where only the slow tier is capacity-scaled — but note the break flagged in
`CODE_MAP.md`: there is no promotion, no demotion, no miss path, and the tiers are not
numerically interchangeable. "Just widen the SWA windows to test long context" is not a valid
experiment, because those layers' RoPE was never trained to reach.

**5. Half of each global head is NoPE.** `partial_rotary_factor: 0.5` means
`rotary_dim = 64`, and `modeling_laguna.py:296-297` splits
`q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]`, leaving `q_pass` — 64 of 128
channels — completely unrotated and concatenated straight back. Those channels carry no
positional signal whatsoever. Laguna's global layers are a **per-channel RoPE/NoPE hybrid**
at exactly 50/50. The model card's phrase is "per-layer-type rotary scales"; the NoPE half
has to be worked out from `partial_rotary_factor`.

**KV arithmetic, corrected.** `num_key_value_heads: 8` is uniform across all layers —
`k_proj`/`v_proj` read the top-level field (`modeling_laguna.py:377-378`) while only the
*query* head count is per-layer (`:359`). So `ASSUMPTIONS.md → kv-per-token-laguna`'s
"treat as an upper bound until recomputed per layer" caveat does not bite for KV: the
per-layer KV cost is exactly `2 · 8 · 128 · 2 B = 4 KiB/token`, everywhere.

| | KV cost |
|---|---|
| 12 global layers | 48 KiB/token, grows with context |
| 36 windowed layers | 4 KiB × 512 × 36 = **72 MiB total, constant** |
| **at 128k context** | 6.0 GiB + 72 MiB |
| **at the advertised 1,048,576** | **48.0 GiB + 72 MiB**, one sequence, bf16, uncompressed |
| hypothetical all-global | 192 GiB at 1M |

`[M]` computed 2026-07-26. Against our measured 62 GiB fast tier
(`[M] notebook/uma-carveout-controls-fast-tier.md`), a single 1M-token Laguna KV cache
nearly fills the fast tier before any weights are resident. The hybrid buys 4×, and it is
the difference between "impossible" and "barely".

## 5. NoPE and length generalization

`[C]` arXiv 2305.19466 (2023) ran the controlled head-to-head — APE, T5-relative, ALiBi,
RoPE, and **no positional encoding at all** — on length generalization. The useful result is
negative: RoPE and ALiBi are not chosen because they extrapolate well, and NoPE does at
least as well as the schemes designed for the job. A causal decoder leaks position for free,
because token `m` can attend to exactly `m+1` keys and nothing else; the count is the signal.

`[C]` arXiv 2404.12224 (2024) shows NoPE also has a ceiling, and attributes it to
"distraction of attention distributions" — fixable by tuning per-head attention
**temperature**, which is the same lever YaRN's `attention_factor` pulls globally and
blindly. `[C]` arXiv 2504.08719 (SWAN-GPT, Apr 2025) builds the architecture Laguna
approximates: interleaved NoPE global layers and RoPE sliding-window layers, reported to
generalize robustly past training length. `[C]` arXiv 2512.12167 (DroPE, Dec 2025) goes
further — *drop* positional embeddings from an already-pretrained model in a recalibration
phase, and extend zero-shot, reportedly beating established RoPE-scaling methods.

**Systems bridge.** NoPE is soft state derived from the causal mask rather than hard state
carried in the payload — the difference between inferring ordering from a monotonically
growing set and stamping a sequence number on every record. **Where it breaks:** a sequence
number survives reordering and loss; the causal-mask signal is only a *count*, so it degrades
into an increasingly weak prior as the count grows. That is precisely the "distraction"
failure 2404.12224 measures.

## 6. Extension training stages

The canonical shape is three stages, not one.

1. **Base pretrain at a short window.** Laguna's is 8192 `[M]`. This is the only stage the
   artifact tells you about; nothing in the config records the stage-2 length or token budget.
2. **Continued pretraining at long window, with the positional scheme changed.** `[C]` ProLong
   (arXiv 2410.02660): Llama-3-8B, 40B tokens total, 64K then 512K sequence length. Long data
   from code repositories and books, but mixed with high-quality short data or general ability
   regresses. **Train beyond your evaluation length.** They explicitly reject perplexity and
   bare needle-in-a-haystack as progress signals — the standard our own ablations should meet.
3. **Short-instruction SFT suffices.** ProLong's counterintuitive finding: SFT on ordinary
   short instruction data was enough; synthetic long-context augmentation was not needed.

Two 2026-era complications. `[C]` SkyLadder (arXiv 2503.15450, NeurIPS 2025, rev. Dec 2025)
finds that under a *fixed token budget* models pretrained with **shorter** context windows
beat their long-context counterparts, and schedules the window upward during pretraining for
up to 3.7% benchmark gain and 22% faster training at 1B/3B scale over 100B tokens — which
makes stage 1 a schedule, not a constant. `[C]` Randomized YaRN (arXiv 2606.23687, Jun 2026)
moves the fix into training: sample YaRN position encodings from a *larger* range than the
training sequences actually span, plus a length curriculum, training entirely under 8K and
improving 16K–128K results on BABILong and MRCR.

**Systems bridge.** This is a rolling upgrade with a canary, not a config flag: you change the
addressing scheme, re-warm on representative traffic, and validate at a length beyond the one
you intend to serve. **Where it breaks:** there is no rollback. The extension overwrites the
weights, and a model that lost short-context ability during stage 2 does not report a
degraded health check — it reports fine on perplexity.

## 7. Effective vs advertised context

`[C]` **RULER** (arXiv 2404.06654, 2024) is the methodology that started this: 13 synthetic
task families across retrieval, multi-hop tracing, aggregation and QA, at controlled lengths.
17 models, all claiming ≥32K. Nearly all score near-perfectly on vanilla NIAH and **only half
maintain satisfactory performance at 32K**. RULER defines *effective context length* as the
longest length at which a model still exceeds **Llama-2-7B's score of 85.6 at 4K**. Read from
the v3 HTML this session:

| model | claimed | effective (RULER) |
|---|---|---|
| GPT-4 | 128K | 64K |
| Llama-3.1-70B | 128K | 64K |
| Command-R-plus | 128K | 32K |
| Qwen2-72B | 128K | 32K |
| Yi-34B | 200K | 32K |
| Mixtral-8x22B | 64K | 32K |
| Gemini-1.5-Pro | 1M | >128K |

The gap did not close. `[C]` **LongBench Pro** (arXiv 2601.02872, Jan 2026): 46 models,
8k–256k, *naturally occurring* rather than synthetic tasks, effective context still typically
shorter than claimed, with pronounced cross-lingual misalignment. `[C]` **ATLAS**
(arXiv 2605.28079, May 2026): 26 models on an 8K–1M grid, and the finding that matters most
for reporting discipline — **7 models shift by ≥2 rank positions** between the 8K–128K regime
and the 8K–1M regime, with individual gaps up to **12 positions**, and the two taxonomy layers
share only **61% of cross-model variance**. A single headline long-context score is not merely
imprecise; the *ordering* it induces is unstable.

**Systems bridge.** Nameplate versus sustained. Advertised context is the sequential-read
number on the box; effective context is QD1 4K random with the write cache disabled. The
uncomfortable part for a benchmark-reader is that NIAH is the sequential-read test: nearly
every model passes it and it predicts almost nothing.

### Five mechanisms behind the gap

1. **Position bias.** `[C]` 2307.03172 (Lost in the Middle): accuracy is highest at the start
   and end of the input and degrades sharply in the middle. Refined by `[C]` 2508.07479
   (Aug 2025): the U-shape holds only up to roughly **50% context occupancy**; above that,
   primacy weakens while recency stays stable and the bias becomes distance-based. So papers
   reporting a clean U and papers reporting recency dominance may just be probing different
   occupancy regimes.
2. **Attention dilution.** Softmax must allocate exactly 1.0 across all keys — there is no way
   to say "none of these are relevant". `[C]` 2506.16640 (ICLR 2026) replaces softmax with
   α-entmax to get genuinely sparse attention, reporting 1000× length extrapolation on
   synthetic tasks and retrieval accuracy at 8× training length. This is a mandatory allocator
   with no admission control and no backpressure — the closest systems analogue is a weighted
   scheduler that cannot idle.
3. **Untrained phase, per §2.** The slow dials have never seen those angles.
4. **Numerics.** `[C]` 2411.13476: bf16 breaks RoPE's relative-position property, and the
   deviation **accumulates as context length increases**, with the first token contributing
   disproportionately. Laguna's implementation partly defends against this — the frequency
   matmul is forced to fp32 (`modeling_laguna.py:118-123`) — but `cos`/`sin` are cast back to
   the model dtype immediately after.
5. **Capacity.** 48 GiB of KV for one 1M-token sequence, per §4. Not a quality failure, but it
   determines what can be run at all.

## Contested — do not let this note read as settled

- **Is position bias architectural or correctable?** `[C]` 2602.16837 (Feb 2026, rev. May 2026)
  derives U-shaped influence profiles from causal masking plus residual connections via
  residual-aware cumulative attention rollout — i.e. it falls out of the architecture and more
  long-context training will not remove it. `[C]` 2406.16008 ("Found in the Middle") recovers
  **up to 15 percentage points** by calibrating the attention bias at inference, which implies
  it is substantially correctable. Both are live.
- **Is rescaling RoPE even the right frame?** Jet-Long `[C]` 2607.07740 (Jul 2026) says keep
  scaling but make it dynamic and length-adaptive (+4.79/+2.18/+2.03 pp over the strongest
  baseline at 1.7B/4B/8B). DroPE `[C]` 2512.12167 says delete the positional embeddings
  instead. SWAN-GPT `[C]` 2504.08719 says design them out from the start. Three incompatible
  answers, all from the last eight months.
- **Architecture or training schedule?** 2506.16640 is an architecture answer to length
  generalization, 2503.15450 and 2606.23687 are training-schedule answers, 2512.12167 is a
  post-hoc surgery answer. The field has not adjudicated, and the resolution probably depends
  on token budget — an axis a small rig can actually attack.
- **Is "effective context" a single number?** ATLAS's rank instability says the induced
  ordering changes with the length grid, which undermines the whole one-number framing that
  RULER's threshold rule encourages.
- **Inherited from `hybrid-architectures`:** whether the SWA/global ratio sets a capability
  ceiling (`[C]` 2507.06457) or only the *rate* at which long-context ability emerges
  (`[C]` 2606.15378). Directly relevant here, because Laguna's entire long reach lives in
  12 layers.
- **Not on arXiv:** Chroma Research's "Context Rot" (Hong, Troynikov, Huber, Jul 2025) is
  widely cited in this track and its claim — that coherent, well-structured input degrades
  attention *more* than shuffled input — is worth knowing. It is an industry tech report with
  no arXiv id. Cite it as such or not at all.

## Open questions

Testable at 20M–300M params, single GPU, ≥62 GiB fast tier `[M]`, no working multi-GPU, and
subject to the `large-tensor-fault-32gib` constraint (keep any single buffer under 32 GiB).

1. **Which half of YaRN does the work — the ramp or the temperature?** Train a 50M model at
   `L = 1024`, extend to 8192 under (a) PI, (b) NTK base change, (c) full YaRN, (d) YaRN with
   `attention_factor` forced to 1.0, (e) NoPE. Matched tokens, ≥3 seeds, scaled-down RULER.
   Nobody publishes (d), and it is the attribution question.
2. **Is `partial_rotary_factor = 0.5` an optimum or an inheritance?** Sweep
   `{1.0, 0.5, 0.25, 0.0}` on the global layers of a 3:1 SWA/global hybrid at 100M. Directly
   probes the per-channel NoPE hypothesis on the exact configuration our reference model ships.
3. **Do `β_fast = 32` / `β_slow = 1` survive re-derivation at a different `θ` and `L_orig`?**
   They are defaults from a 2023 paper applied at `s = 128`. A 2-D sweep at 50M is a few
   GPU-hours.
4. **Does the effective/advertised gap even reproduce below 300M params?** Build a scaled RULER
   (lengths 1×–32× training length, matched short-context baseline for the threshold). If the
   gap does not appear at this scale, this rig cannot study the phenomenon — and that is a
   publishable negative for our own methodology.
5. **bf16 RoPE drift on gfx1151.** Measure `⟨R_m q, R_n k⟩ − ⟨q, R_{n−m} k⟩` in bf16 vs an fp32
   reference as a function of absolute `m` at fixed `m−n`, out to 1M positions. No training
   required, runs in minutes, attacks `ASSUMPTIONS.md → bf16-numerics-unproven` directly, and
   2411.13476 predicts a length-dependent failure that a fixed-shape numerics test would miss.
   **This should run before any long-context training on this machine.**
6. **Is attention temperature one knob or three?** YaRN's `attention_factor`, α-entmax's
   learnable temperature, and 2404.12224's per-head temperature tuning may be the same lever.
   Train NoPE and RoPE arms at 100M, then sweep a single inference-time logit temperature and
   see whether the length-generalization curves collapse onto each other.
7. **Does SkyLadder's short-window-first result hold at 100M?** It was measured at 1B/3B over
   100B tokens. Fixed 2B-token budget, window fixed at 4096 vs scheduled 512→4096.
8. **Where is the KV capacity cliff on our hardware?** At what context length does a
   Laguna-shaped 3:1 hybrid at our scale cross the 62 GiB fast tier, and does crossing it
   produce a throughput cliff? `gpu-fast-tier-size` `[M]` measured ~200 GB/s inside the tier and
   61 GB/s outside it at a 16 GiB carve-out. This is a memory-hierarchy experiment the Z13 is
   unusually well-suited to run, and nobody else is running it.

## Sources

**Read from the repo's own artifacts** (`[M]`, revision pinned in `research/reference/PROVENANCE.md`):

- `research/reference/models/laguna-s/config.json` — `max_position_embeddings: 1048576` (:17),
  `sliding_window: 512` (:41), `rope_parameters` split by layer type (:42-58),
  `num_key_value_heads: 8` (:15), `num_attention_heads_per_layer` (:211-260).
- `research/reference/models/laguna-s/modeling_laguna.py` — `attention_scaling` folded into
  cos/sin under forced fp32 (:118-123); rotary/pass channel split (:295-305); uniform KV-head
  projections vs per-layer query heads (:359, :377-378); separate `swa_rotary_emb` (:607) and
  per-layer-type dispatch (:664-671).
- `research/reference/architecture/transformers/src/transformers/modeling_rope_utils.py` —
  `_compute_yarn_parameters` (:327), `find_correction_dim` (:423), `linear_ramp_factor` (:436),
  correction range (:451), extrapolation/interpolation blend (:454-458).
- `research/reference/architecture/llama-cpp-laguna/src/models/laguna.cpp:184` — SWA layers run
  plain RoPE with YaRN ext/beta forced to zero.
- `ASSUMPTIONS.md` — `gpu-fast-tier-size` (≥62 GiB at ~200 GB/s), `large-tensor-fault-32gib`,
  `bf16-numerics-unproven`, `kv-per-token-laguna`.

**Cited** (`[C]`; arXiv ids verified against arxiv.org this session unless noted):

- arXiv 2104.09864 — RoFormer: Enhanced Transformer with Rotary Position Embedding (2021).
- arXiv 2306.15595 — Extending Context Window of LLMs via Positional Interpolation (Jun 2023).
- arXiv 2309.00071 — YaRN: Efficient Context Window Extension of Large Language Models (2023).
- arXiv 2305.19466 — The Impact of Positional Encoding on Length Generalization in Transformers (2023).
- arXiv 2307.03172 — Lost in the Middle: How Language Models Use Long Contexts (2023).
- arXiv 2404.06654 — RULER: What's the Real Context Size of Your Long-Context Language Models? (2024).
- arXiv 2404.12224 — Length Generalization of Causal Transformers without Position Encoding (Apr 2024).
- arXiv 2406.16008 — Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization (Jun 2024).
- arXiv 2410.02660 — How to Train Long-Context Language Models (Effectively) (Oct 2024).
- arXiv 2411.13476 — When Precision Meets Position: BFloat16 Breaks Down RoPE in Long-Context Training (Nov 2024).
- arXiv 2503.15450 — SkyLadder: Better and Faster Pretraining via Context Window Scheduling (Mar 2025, rev. Dec 2025).
- arXiv 2504.08719 — SWAN-GPT: An Efficient and Scalable Approach for Long-Context Language Modeling (Apr 2025).
- arXiv 2506.16640 — Long-Context Generalization with Sparse Attention (Jun 2025, rev. Mar 2026; ICLR 2026).
- arXiv 2507.06457 — A Systematic Analysis of Hybrid Linear Attention (2025).
- arXiv 2508.07479 — Positional Biases Shift as Inputs Approach Context Window Limits (Aug 2025).
- arXiv 2512.12167 — Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings (Dec 2025).
- arXiv 2601.02872 — LongBench Pro: A More Realistic and Comprehensive Bilingual Long-Context Evaluation Benchmark (Jan 2026).
- arXiv 2602.16837 — A Structural Theory of Position Bias in Transformers (Feb 2026, rev. May 2026).
- arXiv 2605.28079 — ATLAS: All-round Testing of Long-context Abilities across Scales (May 2026).
- arXiv 2606.15378 — Rethinking the Role of Efficient Attention in Hybrid Architectures (Jun 2026).
- arXiv 2606.23687 — Randomized YaRN Improves Length Generalization for Long-Context Reasoning (Jun 2026).
- arXiv 2607.07740 — Jet-Long: Efficient Long-Context Extension with Dynamic Bifocal RoPE (Jul 2026).
- Chroma Research, "Context Rot: How Increasing Input Tokens Impacts LLM Performance" (Hong,
  Troynikov, Huber, Jul 2025) — **industry tech report, no arXiv id**; cited for its claim only.
