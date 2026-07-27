---
title: Reading list — must / should / could, organised to be read alongside the modules
version: 1.0.0
date: 2026-07-26
owner: curriculum-author
status: live document — new findings get folded back in
---

# Reading list

`[G1 — Structured problem solving]`

**Answer first.** There are 76 anchor papers in `research/reference/papers/README.md`, and
this list places every one of them next to the module it explains, adds the tracks the
anchors do not cover (D, E, F), and ranks the lot. But the highest-value reading in this
repo is **not a paper**. It is the thirteen source walkthroughs in
`research/reference/CODE_MAP.md` and one 262-line JSON file, and if you have four hours
this week you should spend them there. The papers tell you what people claim; the code
tells you what shipped. Where they disagree — and they disagree in at least five places
documented below — the disagreement is the finding.

**How this list is ordered.** By curriculum track, then by module inside the track, so you
can read a paper the same evening you read the module that needs it. It is not ordered by
topic in the abstract and it is not a bibliography. Each entry gives you *the specific
thing that paper hands you*, not a summary. If you cannot tell from the line why you would
open the PDF, the line is defective — tell me and I will rewrite it.

**Ranking, honestly.**

| Rank | What it means | Budget |
|---|---|---|
| **must** | The module leans on it. Reading the module without it leaves a hole you will hit later. | Read properly: 45–90 min each. |
| **should** | Removes a specific confusion or supplies a number you will otherwise take on faith. | 20–40 min each; several are skimmable to one figure. |
| **could** | Read when the question comes up, not before. Several exist only so you know the counter-argument exists. | 10–20 min, abstract + one section. |

**Volume warning.** There are 266 entries here: 243 paper entries across the six tracks
(236 distinct arXiv ids — a few earn a place in two tracks) plus 23 non-paper entries.
Ranked, that is 43 **must**, 89 **should**, 111 **could**. At a real 8 hours a week
against a demanding job, the must tier alone is roughly seven weeks if you read nothing
else. That is the wrong plan. The right plan is: five papers now (below), then the
must tier of whichever track you are in, then everything else on demand. `[A]` This list
is a *lookup table with an ordering*, not a queue.

**Verification.** Every arXiv id below resolved against the live arXiv API on
2026-07-26 — 76 from `research/reference/papers/anchors.bib`, 147 from
`curriculum/citation-verification.json`, and 91 newly checked for this file because the
D/E/F modules had never been through the verifier. Every `file:line` below was opened and
the line read before it was written down. No id here is from memory.

---

## If you only read five things

Not five foundational papers. You will learn foundations faster from the modules and from
`training/nanogpt/model.py` than from 2017-era prose, and there is no entry from Track A
in this five for exactly that reason. These are the five that change how you *think about
the problem this lab exists to attack*.

| # | Paper | Why this one |
|---|---|---|
| 1 | **Fast Transformer Decoding: One Write-Head is All You Need** — `arXiv:1911.02150` | Nine pages, 2019, and it is the mental model everything else in this repo is built on: autoregressive decode is bandwidth-bound, not FLOPS-bound. Ranked *should* in the anchors; I am promoting it to first because for a storage engineer it is the single cheapest unlock on the list. Read §2 and stop. |
| 2 | **Efficient Memory Management for LLM Serving with PagedAttention** — `arXiv:2309.06180` | The OS analogy stated by the people who implemented it, and the source of the vocabulary (block table, internal vs external fragmentation, copy-on-write) that every later serving paper assumes you already speak. Then go read where it is *wrong* — `memory/vllm/vllm/v1/core/block_pool.py:679`. |
| 3 | **H₂O: Heavy-Hitter Oracle for Efficient Generative Inference** — `arXiv:2306.14048` | The baseline every eviction policy since is measured against, and the one your pluggable-policy interface has to be able to express or it is not an interface. Read it for the accumulated-attention score, not the theory. |
| 4 | **The Pitfalls of KV Cache Compression** — `arXiv:2510.00231` | This is the lab's thesis in someone else's paper: five published policies pass LongBench while silently dropping specific instructions, with system-prompt leakage as the worked case. If you read one paper that justifies building an attribution instrument instead of a thirty-first eviction policy, it is this one. |
| 5 | **What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction** — `arXiv:2607.08032` | July 2026, the newest thing here, and the first serious argument that KV eviction, prompt compression, recurrent-state bounding and agent-memory consolidation are one problem under a budget. It gives Mnemosyne a single vocabulary for all four legs. Too new to be settled — read it as a proposal, not a result. |

**The sixth, cut for space, and why it hurts:** *Transformers are SSMs* (`arXiv:2405.21060`)
collapses "KV cache versus fixed state" from two architectures into one dial. It is the
best idea on this list. It is cut from the five only because it costs an afternoon and the
other five cost an evening between them.

---

## The non-paper reading, which matters more

### The code — `research/reference/CODE_MAP.md`

Thirteen guided walkthroughs, each with machine-verified `file:line` pointers and a
"where the analogy breaks" paragraph written for your background specifically. Clones are
gitignored; run `scripts/fetch_reference.sh` first. Read in this order — it is not the
order in the file, it is the order that builds.

| Rank | Where | Open at | What you get that no paper gives you |
|---|---|---|---|
| **must** | nanoGPT | `training/nanogpt/config/train_shakespeare_char.py:22`, `training/nanogpt/README.md:51` | The entire model surface is six dataclass fields and the pass/fail number is `1.4697`. This is the Hardware Validation Gate's known-good run and the only whole-system artifact small enough to hold in your head. |
| **must** | Laguna in `transformers` | `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:365` | The whole SWA/global hybrid is one list lookup: `config.layer_types[layer_idx]`. Every hybrid-ratio question this lab asks reduces to what is in that list — which makes the research question trivially ablatable and much less mystical than the papers make it sound. |
| **must** | vLLM prefix cache | `memory/vllm/vllm/v1/core/kv_cache_utils.py:596`, `memory/vllm/vllm/v1/core/block_pool.py:679` | The cache key is a *chain* (parent hash folded in), so it is position-dependent and strictly prefix-ordered — one changed token at position 0 invalidates every downstream hash. And freeing is not evicting: `_maybe_evict_cached_block` runs lazily at reallocation, so "blocks in use" and "entries available for hits" are two different numbers. Neither fact is in the paper. |
| **should** | SGLang RadixAttention | `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:565`, `memory/sglang/python/sglang/srt/mem_cache/evict_policy.py:16` | Eviction is topologically constrained, not recency-ordered — only leaves are candidates, so a hot child pins a cold parent forever. And the entire replacement-policy surface is one `get_priority(node)` function in a 65-line file. That file is the shape Mnemosyne's policy interface should be. |
| **should** | Mooncake Store | `memory/mooncake/mooncake-store/src/master_service.cpp:6382`, `memory/mooncake/mooncake-store/include/replica_selection.h:122` | A KV cache that became a real distributed store: leases instead of LRU, a fixed tier preference ladder instead of a cost model, and TinyLFU-gated promotion. Also the cleanest statement of the break — evicting KV is never data loss, only recompute, so it can throw bytes away rather than block on writeback. No storage tier you have run is allowed to do that. |
| **should** | Mamba-2 SSD | `architecture/mamba/mamba_ssm/modules/ssd_minimal.py:34`, `architecture/mamba/mamba_ssm/modules/mamba2.py:352` | Forty lines of pure PyTorch that are the whole constant-state idea, plus proof that the inference state has no `seqlen` dimension. Read `ssd_minimal` *before* the Triton kernels or you will lose a day. |
| **should** | Gated DeltaNet | `architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:54` and `architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:56` | Two adjacent lines: an indiscriminate global decay, then a read-before-write that erases exactly one direction. The mental model "gating = selective forgetting" is backwards, and you can see it is backwards in six lines of code. |
| **should** | OLMo-core training step | `training/olmo-core/src/olmo_core/train/trainer.py:1037` and `training/olmo-core/src/olmo_core/train/trainer.py:1394` | Telemetry as unevaluated device tensors drained every N steps — batched not for I/O amortisation but to avoid a host-device sync. Observability costs training throughput here, which is not true of any logging system you have operated. |
| **could** | Laguna in llama.cpp | `architecture/llama-cpp-laguna/src/llama-kv-cache-iswa.cpp:73`, `architecture/llama-cpp-laguna/src/models/laguna.cpp:184` | Two entirely separate KV cache objects, the SWA one sized to `n_swa + n_ubatch` rather than to context. Read it for the sizing math and for the fact that the two tiers apply *different RoPE* — you cannot "just widen the windows" to test long context. |
| **could** | FlashInfer paged KV | `memory/flashinfer/flashinfer/page.py:403`, `memory/flashinfer/flashinfer/decode.py:1239` | NHD vs HND is the stride order inside a page and it is a physical kernel requirement, not a view. Also: `plan()` is a query planner, not an MMU — the page table is frozen for a whole forward pass. |
| **could** | Samba | `architecture/samba/lit_gpt/model.py:323` | The layer schedule is not stored, it is recomputed from `layer_idx % mb_per_layer` — striping, not a map. Read it to see how little machinery a hybrid actually needs. |
| **could** | Checkpoint sharding and mid-epoch resume | `training/olmo-core/src/olmo_core/distributed/checkpoint/__init__.py:702`, `training/olmo-core/src/olmo_core/train/checkpoint.py:498`, `training/olmo-core/src/olmo_core/data/data_loader.py:667` | Optimizer state keyed by parameter FQN, so resharding is re-indexing rather than data movement; atomicity is a directory rename, so a torn save loses everything rather than a tail. And the dataloader is not a WAL — it stores a cursor and re-derives the whole epoch permutation as a pure function of (seed, epoch, length, chunk size), so restart costs a recomputation rather than a replay. DR with no journal and no incremental delta. |
| **could** | vLLM block table | `memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:97`, `memory/vllm/vllm/v1/attention/ops/triton_unified_attention.py:424` | A software page walk performed inside the attention kernel, once per KV tile, with no MMU and no TLB. Read the Triton line to see what translation actually costs. |

### Laguna's own config — the shortest high-value read in the repo

`research/reference/models/laguna-s/config.json` is 262 lines and settles more architecture
questions than any paper on this list. Fetched at revision `b0a9fd7c850e` (see
`research/reference/PROVENANCE.md`). **must** — twenty minutes, and take notes.

| Open at | What it settles |
|---|---|
| `models/laguna-s/config.json:59` | `layer_types` — 48 entries, 12 `full_attention` + 36 `sliding_attention` in a strict GSSS pattern. `[M]` The 3:1 hybrid ratio, read from the shipped artifact rather than quoted from a paper (`ASSUMPTIONS.md → reference-model`). Four labs shipped 3:1; none ablated it. This is the file that turns folklore into a measurement. |
| `models/laguna-s/config.json:41` | `sliding_window: 512`. Small. Much smaller than the hybrid papers condition you to expect, and it is what makes 36 of 48 layers cost O(512) KV per token. |
| `models/laguna-s/config.json:42` | `rope_parameters` — **two** schedules, one per layer type. Full layers get YaRN at θ=500000 over half the head dims; sliding layers get plain RoPE at θ=10000 over all of them. The two layer types are not the same block with a different mask. |
| `models/laguna-s/config.json:211` | `num_attention_heads_per_layer` — 48 query heads on full layers, 72 on sliding. `[M]` The top-level `num_attention_heads: 48` is wrong for 36 of 48 layers (`ASSUMPTIONS.md → laguna-heads-uniform`). What varies is the GQA group size G: 6 on full, 9 on sliding. `num_key_value_heads` stays 8, so KV cost is still exact: `[M]` **192.0 KiB/token**, 24.0 GiB at 128k context, against a `[M]` ≥62 GiB fast tier. |
| `models/laguna-s/config.json:26` | `router_aux_loss_coef: 0.0` — the aux-loss-free load-balancing claim, confirmed in the shipped config rather than inferred from the paper. Pair with `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:185`. |
| `models/laguna-xs/config.json:17` | `max_position_embeddings: 262144` on XS versus 1048576 on S, from the same `original_max_position_embeddings: 8192` — 8192 × 32 and 8192 × 128. `[M]` The "1M context" number is pretraining length times YaRN factor, exactly (`research/synthesis.md → Folklore`). Inherited convention, not demonstrated capability. |

### The lab's own documents, which supersede papers where they conflict

**must**, in this order, before you start any track:

- `research/synthesis.md` — what we believe and what we refuse to touch, with the MECE issue tree and the five questions worth our compute. Read the "Folklore" and "What is weakest in our own evidence" sections twice.
- `ASSUMPTIONS.md` — the only place a `[M]` number is allowed to originate. If a module quotes a hardware number that is not in a row here, treat it as unregistered.
- `research/reference/CODE_MAP.md` — see above.
- `research/memory/open-problems-ranked.md` — the P/T/E scoring that says attribution is the only 5/5/5.

**should:** the nine other survey notes in `research/memory/`, each of which is the long
form of one Track C module and mirrors it 1:1; `research/notes/evaluation-landscape.md`
before any of Track E; `notebook/uma-carveout-controls-fast-tier.md` as the worked example
of the house hypothesis-card format.

---

## Track A — Foundations

Six modules: `tensors-and-autograd`, `transformer-forward-pass-by-hand`, `tokenization`,
`the-training-loop`, `loss-and-optimization`, `scaling-laws-and-flops-budget`.

`[A]` The papers here are the *least* load-bearing on the list. The modules plus nanoGPT
source teach this material better than the originals do, because the originals are arguing
with a 2017 audience you are not. Read the must tier; treat the rest as reference.

### must

- **Attention Is All You Need** — `arXiv:1706.03762` — *with* `transformer-forward-pass-by-hand`. Read it for what is *absent*: no RMSNorm, no RoPE, no GQA, no SwiGLU, pre-LN not yet invented. Every later module is a diff against this document, and knowing the diff is most of Track B.
- **Scaling Laws for Neural Language Models** — `arXiv:2001.08361` — *with* `scaling-laws-and-flops-budget`. Read it for the 6·N·D compute accounting and the power-law form, then read the two corrections below before you believe any of its allocation advice.
- **Training Compute-Optimal Large Language Models** — `arXiv:2203.15556` — *with* `scaling-laws-and-flops-budget`. Read it for the three IsoFLOP estimation approaches and the discipline they impose: params-vs-tokens allocation is stated before an arm runs, never defended afterwards. This is the methodological backbone of Themis.
- **Adam: A Method for Stochastic Optimization** — `arXiv:1412.6980` — *with* `loss-and-optimization`. Read it for the two moment buffers, because those buffers are 2× your parameter memory and they are why a 300M model does not train in 600 MB.
- **Decoupled Weight Decay Regularization** — `arXiv:1711.05101` — *with* `loss-and-optimization`. Read it for the one-line difference between Adam and AdamW and why the L2-in-the-gradient version silently couples decay to the learning rate.
- **Neural Machine Translation of Rare Words with Subword Units** — `arXiv:1508.07909` — *with* `tokenization`. Read it for the merge algorithm itself; everything about modern tokenizers is this plus engineering.
- **FlashAttention** — `arXiv:2205.14135` — *with* `tensors-and-autograd` and `attention-variants-and-kv-cost`. Read it for the IO-complexity argument — the score matrix never needs to exist. Then note `[M]` that on gfx1151 by default **it still does**: 147.2 bytes/T² retained versus 6.6 with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, and `flash_sdp_enabled()` returns True either way (`ASSUMPTIONS.md → sdpa-is-memory-efficient`). The paper is right; your machine is not running it.

### should

- **Automatic differentiation in machine learning: a survey** — `arXiv:1502.05767` — *with* `tensors-and-autograd`. Read this for the forward-mode/reverse-mode distinction and why reverse mode costs one graph's worth of retained activations — the first place memory bites.
- **Training Deep Nets with Sublinear Memory Cost** — `arXiv:1604.06174` — *with* `the-training-loop`. Read it for the √n checkpointing tradeoff stated as arithmetic: recompute is a currency you can spend against activation memory, at a known exchange rate.
- **Mixed Precision Training** — `arXiv:1710.03740` — *with* `the-training-loop`. Read it for loss scaling and the fp32 master-weight copy, both of which are still in every trainer you will read.
- **An Empirical Model of Large-Batch Training** — `arXiv:1812.06162` — *with* `the-training-loop`. Read it for the gradient-noise-scale definition of critical batch size — the one principled answer to "is my batch too small," and cheap to measure at our scale.
- **On the difficulty of training Recurrent Neural Networks** — `arXiv:1211.5063` — *with* `loss-and-optimization`. Read it only for the gradient-clipping section: clipping is a spike suppressor, not a regulariser, and this is where that is argued.
- **Language Models are Few-Shot Learners** — `arXiv:2005.14165` — *with* `scaling-laws-and-flops-budget`. Read the appendix hyperparameter table, not the paper. It is the last time a frontier lab published a complete recipe.
- **Reconciling Kaplan and Chinchilla Scaling Laws** — `arXiv:2406.12907` — *with* `scaling-laws-and-flops-budget`. Read this for the resolution: the two laws disagree because one counts embedding parameters and the other does not. A units bug, which is exactly the kind of thing that will bite our own fits.
- **Chinchilla Scaling: A replication attempt** — `arXiv:2404.10102` — *with* `scaling-laws-and-flops-budget`. Read it for what it feels like when a famous fit does not reproduce, and for how few points the original had.
- **Scaling Laws with Vocabulary** — `arXiv:2407.13623` — *with* `tokenization`. Read this for the result that vocabulary size is a scaling axis with its own optimum, which turns "which tokenizer" from taste into budget.
- **Understanding Warmup-Stable-Decay Learning Rates** — `arXiv:2410.05192` — *with* `loss-and-optimization`. Read it for the river-valley picture that explains why WSD's cooldown works and why you can branch a run at the end of the stable phase — directly useful for cheap ablations.
- **Beyond Chinchilla-Optimal: Accounting for Inference** — `arXiv:2401.00448` — *with* `scaling-laws-and-flops-budget`. Read it for the reframing that a compute-optimal model is the wrong target when you will serve it; smaller and over-trained wins on total cost.
- **Fishing for Magikarp** — `arXiv:2405.05417` — *with* `tokenization`. Read it for the detection method for under-trained tokens, and because it is the clearest demonstration that the tokenizer and the model can silently disagree about what exists.
- **Tokenization counts: the impact of tokenization on arithmetic** — `arXiv:2402.14903` — *with* `tokenization`. Read it for the concrete failure: digit grouping decides arithmetic accuracy. The cheapest possible demonstration that a preprocessing choice is an architecture choice.
- **Cut Your Losses in Large-Vocabulary Language Models** — `arXiv:2411.09009` — *with* `loss-and-optimization`. Read it for the logits-tensor memory accounting — at a 100k vocab the loss layer, not attention, is your peak allocation.

### could

- **Fantastic Pretraining Optimizers and Where to Find Them** — `arXiv:2509.02046` — *with* `loss-and-optimization`. Read it for the controlled comparison that most claimed optimizer speedups shrink under matched tuning budgets.
- **SOAP, Muon, and Beyond** — `arXiv:2607.20548` — *with* `loss-and-optimization`. Read it for the current state of the AdamW-versus-Muon dispute, which `research/synthesis.md` lists as contested and leaves contested.
- **Adafactor** — `arXiv:1804.04235` — Read it for the factored second-moment trick, i.e. what you give up to halve optimizer state.
- **8-bit Optimizers via Block-wise Quantization** — `arXiv:2110.02861` — Read it for the same tradeoff taken further. Note `bitsandbytes` crashes on import on this stack, so this is design reading only.
- **Byte Latent Transformer** — `arXiv:2412.09871` — *with* `tokenization`. Read it for the strongest current argument that the tokenizer should not exist, and for the entropy-based patching that replaces it.
- **TokSuite** — `arXiv:2512.20757` — *with* `tokenization`. Read it for a systematic measurement of what tokenizer choice actually changes downstream, which is less than the folklore claims and different from where you would guess.
- **Say Anything but This: When Tokenizer Betrays Reasoning in LLMs** — `arXiv:2601.14658` — Read it for a 2026 instance of the same failure class, on reasoning rather than arithmetic.
- **Toward a Theory of Tokenization in LLMs** — `arXiv:2404.08335` — Read it if you want the formal statement of why tokenization is not neutral preprocessing.
- **Understanding and Mitigating Tokenization Bias** — `arXiv:2406.16829` — Read it for the sampling-level bias a BPE boundary introduces, which matters if you ever score token-level probabilities.
- **Tokens-per-Parameter Coverage Is Critical for Robust Scaling Law Extrapolation** — `arXiv:2605.08541` — *with* `scaling-laws-and-flops-budget`. Read it for the condition under which a small-scale fit extrapolates at all — directly relevant to `ablation-scale-sufficient`, which is still untested.
- **Optimal Learning-Rate Schedules under Functional Scaling Laws** — `arXiv:2602.06797` — Read it for the derivation of WSD from a scaling-law argument rather than from folklore.

---

## Track B — Modern architecture

Five modules: `attention-variants-and-kv-cost`, `normalization-and-activations`,
`positional-encoding`, `moe-and-routing`, `depth-width-and-initialization`.

### must

- **GQA: Training Generalized Multi-Query Transformer Models** — `arXiv:2305.13245` — *with* `attention-variants-and-kv-cost`. Read it for the KV-sizing arithmetic and the one knob (H → G → 1) that essentially every shipped open model now sets, plus the uptraining recipe that converts an MHA checkpoint for ~5% of pretrain compute.
- **DeepSeek-V2** — `arXiv:2405.04434` — *with* `attention-variants-and-kv-cost`. Read it for Multi-head Latent Attention and the decoupled-RoPE trick that makes low-rank KV compatible with rotary position. Then note `[M]` the HF reference implementation expands the latent back to full per-head K and V *before* the cache write, so the 93.3% reduction is not what that code does (`research/synthesis.md → Folklore`).
- **RoFormer: Enhanced Transformer with Rotary Position Embedding** — `arXiv:2104.09864` — *with* `positional-encoding`. Read it for the rotation mechanics. Load-bearing for Mnemosyne: the cache stores *post-rotation* keys, so every eviction, compaction or repacking scheme has to answer whether it silently changed a token's encoded position.
- **Root Mean Square Layer Normalization** — `arXiv:1910.07467` — *with* `normalization-and-activations`. Read it for the observation that re-centring does nothing and only re-scaling matters — one of the few simplifications in this field that was purely free.
- **GLU Variants Improve Transformer** — `arXiv:2002.05202` — *with* `normalization-and-activations`. Read it for SwiGLU and for the 2/3 width correction that keeps the parameter count matched. That correction is why FFN widths look like 12288 rather than a round number.
- **DeepSeek-V3 Technical Report** — `arXiv:2412.19437` — *with* `moe-and-routing`. Read it for the complete modern MoE recipe in one document — sigmoid gating, per-expert bias, shared plus fine-grained experts, node-limited routing — and because it is the closest published analogue to Laguna.
- **Auxiliary-Loss-Free Load Balancing for MoE** — `arXiv:2408.15664` — *with* `moe-and-routing`. Read it for the bias-update rule isolated from a model launch, with its hyperparameter and controlled 1B/3B ablations, plus the batch-level versus global-level distinction that explains why aux-loss numbers flatter themselves.
- **Tensor Programs V: Zero-Shot Hyperparameter Transfer** — `arXiv:2203.03466` — *with* `depth-width-and-initialization`. Read it for the reason a 20M–300M ablation can say anything about anything: muP makes the optimal LR stable across width, so an arm comparison stops being a comparison of tuning luck.

### should

- **Fast Transformer Decoding (MQA)** — `arXiv:1911.02150` — *with* `attention-variants-and-kv-cost`. See the five above. Read §2.
- **Query-Key Normalization for Transformers** — `arXiv:2010.04245` — *with* `normalization-and-activations`. Read it for QK-norm, which Laguna applies at head_dim *before* RoPE (`architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:368`) — a stability measure that became standard without ever getting a flagship paper.
- **On Layer Normalization in the Transformer Architecture** — `arXiv:2002.04745` — Read it for the gradient-magnitude argument that killed post-LN and made warmup optional. This is the paper behind every "pre-LN" in every config you will read.
- **YaRN: Efficient Context Window Extension** — `arXiv:2309.00071` — *with* `positional-encoding`. Read it for the frequency-band view of RoPE extension, and because it is the cleanest single place to learn Position Interpolation and NTK-aware scaling too. `[M]` Laguna's `attention_factor` of 1.4852030263919618 matches YaRN's default temperature formula to the last digit — this paper is literally in the config.
- **The Impact of Positional Encoding on Length Generalization** — `arXiv:2305.19466` — *with* `positional-encoding`. Read it for the negative result: RoPE and ALiBi were not chosen for extrapolation, and NoPE beats both in the controlled comparison. Do not assume your positional scheme is what buys you long context.
- **Round and Round We Go! What makes Rotary Positional Encodings useful?** — `arXiv:2410.06205` — Read it for the mechanistic account: the low-frequency bands are used for semantics, not position, which is why naive interpolation hurts where it does.
- **Train Short, Test Long (ALiBi)** — `arXiv:2108.12409` — Read it for the alternative that lost, and for the clean statement of what "length extrapolation" would mean if you had it.
- **Massive Activations in Large Language Models** — `arXiv:2402.17762` — *with* `normalization-and-activations`. Read it for the empirical fact that a handful of hidden dimensions carry enormous magnitude, which is simultaneously the attention-sink mechanism and the reason activation quantization is hard.
- **When Precision Meets Position: BFloat16 Breaks Down RoPE** — `arXiv:2411.13476` — *with* `positional-encoding`. Read this before you trust any long-context number from this machine: RoPE's phase accumulates error in bf16 at large positions, and `bf16-numerics-unproven` is still untested here. `research/synthesis.md` recommends adding exactly this test to the Hardware Validation Gate.
- **ST-MoE: Designing Stable and Transferable Sparse Expert Models** — `arXiv:2202.08906` — *with* `moe-and-routing`. Read it as a systems postmortem: router z-loss, precision-driven divergence, capacity factors, the train/finetune gap. It names most of the ways an MoE run goes wrong quietly.
- **DeepSeekMoE** — `arXiv:2401.06066` — *with* `moe-and-routing`. Read it for fine-grained segmentation plus shared-expert isolation, and for their expert-specialisation measurements — the closest thing to an operational definition of expert collapse.
- **Switch Transformers** — `arXiv:2101.03961` — Read it for top-1 routing and the capacity-factor/token-dropping mechanics, which is the part later papers assume you already know.
- **Outrageously Large Neural Networks (the original MoE)** — `arXiv:1701.06538` — Read it for the load-balancing loss in its original form, so you can see what the aux-loss-free line is actually replacing.
- **Small-scale proxies for large-scale Transformer training instabilities** — `arXiv:2309.14322` — *with* `depth-width-and-initialization`. Read this one carefully. It is the paper that says our scale *can* reproduce the failures that matter, which is the load-bearing assumption of the whole lab.
- **DeepNet: Scaling Transformers to 1,000 Layers** — `arXiv:2203.00555` — Read it for the residual-scaling derivation, i.e. what actually breaks at depth and what the fix costs.
- **Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration** — `arXiv:2512.22382` — *with* `depth-width-and-initialization`. Read it as the practical superset of muP: transfer along four axes, not one, with per-module hyperparameters.
- **μ-Parametrization for Mixture of Experts** — `arXiv:2508.09752` — *with* `moe-and-routing`. Read it because routing and sparsity sit outside classic muP theory, so plain Tensor Programs V is not sufficient for a Proteus MoE arm.

### could

- **Sigmoid Gating is More Sample Efficient than Softmax Gating** — `arXiv:2405.13997` — Read it for the actual argument behind a `proteus-moe-sigmoid` arm. Small-scale regression theory, so it motivates the ablation rather than settling it.
- **When Are Experts Misrouted? Counterfactual Routing Analysis** — `arXiv:2605.07260` — Read it for the evidence that a balanced router is not a good router: the trained top-k choice is near-optimal on confident tokens and near-uninformative on hard ones.
- **The Myth of Expert Specialization in MoEs** — `arXiv:2604.09780` — Read it for the counter-claim that routing reflects geometry rather than domain, which is the direct rebuttal to DeepSeekMoE's specialisation story.
- **Transformers without Normalization** — `arXiv:2503.10622` — Read it for the claim that a tanh-based element-wise op replaces LayerNorm entirely, and decide for yourself whether the ablations support it.
- **Peri-LN: Revisiting Normalization Layer Placement** — `arXiv:2502.02732` — Read it for the third placement option and the variance-growth argument for it.
- **The Curse of Depth in Large Language Models** — `arXiv:2502.05795` — Read it for the measurement that deep layers in pre-LN models contribute little, and the scaling fix proposed.
- **Softpick: No Attention Sink, No Massive Activations** — `arXiv:2504.20966` — Read it for the rectified-softmax variant that removes the sink by construction, which is directly relevant if you ever want an eviction policy that does not need a pinned prefix.
- **Attention Sinks Are Provably Necessary in Softmax Transformers** — `arXiv:2603.11487` — Read it for the opposite conclusion, argued formally. Hold both.
- **Gated Attention for Large Language Models** — `arXiv:2505.06708` — Read it for the mechanism behind Laguna's per-head output gate (`architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:370`), which is not in the standard decoder recipe and has no flagship paper.
- **QK-Normed MLA** — `arXiv:2606.16310` — Read it for how QK-norm and latent KV interact, which is a live problem if Proteus ever combines them.
- **Multi-Head Low-Rank Attention** — `arXiv:2603.02188` — Read it for a 2026 point in the MLA design space with different tradeoffs.
- **Sparse Query Attention** — `arXiv:2510.01817` — Read it for the orthogonal knob: reduce *query* heads rather than KV heads, trading compute rather than cache.
- **AdaRoPE: Not All Attention Heads Should Rotate and Scale Equally** — `arXiv:2607.19363` — Read it for per-head RoPE scaling, which is where the extension literature has gone.
- **The Depth-to-Width Interplay in Self-Attention** — `arXiv:2006.12467` — *with* `depth-width-and-initialization`. Read it for the theoretical depth-efficiency argument and its stated regime of validity.
- **MobileLLM** — `arXiv:2402.14905` — Read it for the empirical finding that deep-and-thin beats wide-and-shallow below ~1B, which is directly our ablation scale.

---

## Track C — Memory (the deep track)

Ten modules, mirroring `research/memory/` 1:1. This track gets the most weight in the
schedule and the most entries here. Papers are listed once, at the module where you first
need them.

### memory-taxonomy-for-engineers

**must**
- **Rethinking Memory in LLM based Agents** — `arXiv:2505.00675` — Read it for the six atomic operations (consolidation, updating, indexing, forgetting, retrieval, compression). That operation vocabulary maps almost 1:1 onto cache and storage primitives you already own, which is why it is the entry point rather than the human-memory surveys.
- **What to Keep, What to Forget: A Rate–Distortion View** — `arXiv:2607.08032` — See the five above. Read it for the seven-axis taxonomy and for the recurring failure it names: every one of these systems discards irreversibly *before the query is known*.

**should**
- **MemOS: A Memory OS for AI System** — `arXiv:2507.03724` — Read it for the parametric/activation/plaintext trichotomy and the MemCube unit with provenance and versioning. Closest thing in the literature to a storage-hierarchy mental model — and read it alongside the module's argument for why the hierarchy analogy stops paying.
- **From Tensor Buffer to Distributed Memory Hierarchy** — `arXiv:2607.02574` — Read it for four axes (locality, lifetime, ownership, substrate) and five archetypes, and above all for the **seven named measurement gaps**. For a lab whose stated deliverable is attribution, that gap list is the most actionable page in the whole reading list.
- **A Survey on LLM Acceleration based on KV Cache Management** — `arXiv:2412.19442` — Read it for the token-level / model-level / system-level partition that most later KV papers position themselves against. Still the better *first* taxonomy even though 2607.02574 is newer.

**could**
- **From Human Memory to AI Memory** — `arXiv:2504.15965` — Read it for the explicit human-term-to-LLM-term mapping, useful precisely for seeing where the cognitive analogy stops paying.
- **A Survey on the Memory Mechanism of LLM based Agents** — `arXiv:2404.13501` — Read it for provenance: the origin-point survey whose design/evaluation split later taxonomies inherit without re-justifying.
- **Contextual Agentic Memory is a Memo, Not True Memory** — `arXiv:2604.27707` — Read it for the sharpest statement of the contested question this module refuses to settle: whether any of this is memory at all, or externalised note-taking.

### kv-cache-mechanics

**should**
- **DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence** — `arXiv:2606.19348` — Read it for the axis change: Compressed Sparse Attention folds *m tokens into one KV entry*, compressing along sequence rather than along feature. MLA shrinks each entry; this shrinks how many exist. A 2026 KV-budget experiment that varies only one of those axes is under-designed.
- **Towards Efficient LLM Serving: A Survey on System-Aware KV Cache Optimization** — `arXiv:2607.08057` — Read it for the temporal/spatial/structural framing, which is the closest the literature comes to speaking your native vocabulary.
- **The Economics of AI Decoding Chips** — `arXiv:2607.13068` — Read it for the compute-versus-capacity-versus-bandwidth rebalancing argument, which is the closest published statement of why the Z13's inverted ratio is a research instrument rather than a compromise.

**could**
- **DeepSeek-V3.2** — `arXiv:2512.02556` — Read it for DeepSeek Sparse Attention layered on top of MLA, the intermediate step between V2 and V4 that makes the two-axis story legible.
- **Towards Economical Inference: Enabling MLA in Any Transformer** — `arXiv:2502.14837` — Read it for the concrete GQA → MLA conversion recipe (partial-RoPE removal plus joint SVD), recoverable on 0.3–0.6% of the data — the cheapest way to run an MLA-versus-GQA ablation without pretraining twice.

### kv-eviction-policies

**must**
- **H₂O: Heavy-Hitter Oracle** — `arXiv:2306.14048` — See the five above.
- **Efficient Streaming Language Models with Attention Sinks** — `arXiv:2309.17453` — Read it for the attention-sink result: evicting the first few tokens collapses the model. This is why every policy since pins a prefix, and it is the first hard constraint your policy interface must express.
- **SnapKV** — `arXiv:2404.14469` — Read it for the observation-window trick — compress at prefill using the last ~32 query positions. PyramidKV, Ada-KV, FastKV and RocketKV are all extensions of this one mechanism, and several groups argue the window is where all the gain actually lives.

**should**
- **PyramidKV** — `arXiv:2406.02069` — Read it for non-uniform per-layer budget allocation and the depth-wise attention-concentration evidence — plus the paper's own admission that it converges to SnapKV as the ratio gets aggressive.
- **KeyDiff** — `arXiv:2504.15364` — Read it for eviction that never reads an attention score. This is the option you need under fused kernels that never materialise the attention matrix — which, `[M]` on this machine with the AOTriton flag set, is the configuration you will actually be running.
- **The Pitfalls of KV Cache Compression** — `arXiv:2510.00231` — See the five above.
- **KV Cache Optimization Strategies for Scalable and Efficient LLM Inference** — `arXiv:2603.20397` — Read it for the March 2026 map: five technique families against seven deployment scenarios, with the explicit finding that no single method dominates. Cite this when someone asks why we are not implementing the winner.

**could**
- **RocketKV** — `arXiv:2502.14051` — Read it for the argument that permanent eviction and dynamic sparse attention are complementary rather than rivals, with decode-phase memory and bandwidth numbers attached.
- **Learning to Evict from Key-Value Cache** — `arXiv:2602.10238` — Read it for the direct analogue of learned cache replacement: lightweight per-head RL agents ranking tokens by predicted future usefulness. The first credible break from attention-mass heuristics.
- **Ada-KV** — `arXiv:2407.11550` — Read it for head-level rather than layer-level budget allocation, the other half of the "is non-uniform allocation real" dispute.
- **A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression** — `arXiv:2406.11430` — Read it as the origin of attention-free scoring, and note that *Pitfalls* finds K-Norm among the methods that silently drop instructions. Cheapness and robustness are in tension here.
- **When Does Value-Aware KV Eviction Help? A Fixed-Contract Diagnostic** — `arXiv:2605.08234` — Read it for the argument that task accuracy alone cannot tell you *why* a selector worked. Methodologically the closest published work to the Mnemosyne deliverable; read it before designing the oracle-diff harness.
- **Taming the Fragility of KV Cache Eviction** — `arXiv:2510.13334` — Read it for the worst-case-versus-mean aggregation result: rankings do not survive the change of aggregator. One of the four documented outcome-held/mechanism-broke cases in `research/synthesis.md`.
- **Error Certificates for KV-Cache Eviction via Randomized Design** — `arXiv:2607.21475` — Read it for the only attempt here at a *bound* rather than a benchmark. If a certificate is achievable at our scale, it changes what `mnemosyne-core` should ship.
- **CompressKV** — `arXiv:2606.24467` — Read it as a representative of the retrieval-guided branch, which treats "what to keep" as a semantic question rather than an attention-statistics one.

### paged-attention-and-prefix-reuse

**must**
- **PagedAttention** — `arXiv:2309.06180` — See the five above.
- **Mooncake** — `arXiv:2407.00079` — Read it for what the KV cache becomes once it is a first-class distributed store: DRAM/SSD tiering across a fleet, cache-aware scheduling, prediction-based early rejection under overload. Production numbers, not a simulator. Then read `memory/mooncake/mooncake-store/src/master_service.cpp:6382` and see that the eviction policy is leases, not LRU.

**should**
- **SGLang** — `arXiv:2312.07104` — Read it for RadixAttention — prefix reuse as an LRU radix tree — and for the moment cache hit rate stops being an implementation detail and becomes a scheduling objective.
- **vAttention** — `arXiv:2405.04437` — Read it as the counter-argument to paging: CUDA VMM gives you defragmentation *and* virtual contiguity, so stock attention kernels work unmodified. The sharpest available test of whether the OS-paging analogy is load-bearing or merely familiar. Your systems instinct ("of course you page") is the thing under dispute.
- **DistServe** — `arXiv:2401.09670` — Read it for the goodput argument — TTFT and TPOT as separate SLOs — and the KV-transfer cost model that decides when the prefill/decode split actually pays.
- **LMCache** — `arXiv:2510.09665` — Read it for the engineering detail of what a tiering layer must actually implement, as a layer *beneath* the engine rather than inside it.

**could**
- **TraCT: CXL Shared Memory KV Cache at Rack-Scale** — `arXiv:2512.18194` — Read it for the synchronisation and consistency problems non-coherent shared memory forces on you — more instructive than the CXL bandwidth story. Note that our unified-memory platform collapses the tier boundary these papers are built around, which is either a confound or a natural experiment.
- **Splitwise** — `arXiv:2311.18677` — Read it for the hardware-heterogeneity angle DistServe leaves out, with per-phase provisioning numbers.
- **Sarathi-Serve** — `arXiv:2403.02310` — Read it for the counter-position to disaggregation: stall-free chunked prefill inside one pool captures most of the benefit without paying KV transfer.

### constant-state-memory

**must**
- **Mamba** — `arXiv:2312.00752` — Read it for input-dependent (selective) gating plus the hardware-aware scan. The systems half is as important as the math half, and it is the half most summaries drop.
- **Zoology: Measuring and Improving Recall in Efficient Language Models** — `arXiv:2312.04927` — Read it for MQAR, which isolates multi-query associative recall and shows the failure is a *state-capacity* limit rather than a training artifact. This is the benchmark your own eviction work should be scored against, because unlike NIAH it has a closed-form capacity prediction you can check your harness against.
- **Transformers are SSMs (SSD)** — `arXiv:2405.21060` — Read it for the unification: attention and SSMs are two decompositions of the same structured matrix, so "KV cache versus fixed state" becomes one dial — state size — rather than two architectures. This is the single best idea on the list.

**should**
- **Simple linear attention language models balance the recall-throughput tradeoff** — `arXiv:2402.18668` — Read it for the explicit recall-versus-state-size Pareto frontier. It treats state as a fixed budget you spend, which is the closest thing in the literature to a capacity-planning argument.
- **Parallelizing Linear Transformers with the Delta Rule** — `arXiv:2406.06484` — Read it for the mechanism everything after 2024 builds on: state as a linear associative memory that gets *corrected* (write-with-erase) instead of blindly accumulated.
- **Gated Delta Networks** — `arXiv:2412.06464` — Read it for the erase/update decomposition, and because it is the layer that actually shipped in Qwen3-Next and Olmo Hybrid — so it is the realistic baseline, not the interesting one.
- **Mamba-3** — `arXiv:2603.15569` — Read it for the current SSM state of the art with an inference-first framing, and for the headline capacity result: Mamba-2 perplexity at half the state size.

**could**
- **Sparse Delta Memory** — `arXiv:2607.07386` — Read it for the most systems-legible attack on the capacity wall: replace the dense outer-product state with sparse addressed reads and writes into a large explicit memory, measured under isoFLOP. This is a memory hierarchy, spelled out.
- **Log-Linear Attention** — `arXiv:2506.04761` — Read it because it directly attacks this track's defining O(1) assumption, replacing the fixed state with a logarithmically growing hierarchy of states. Arguably the most important conceptual reframing of the last year.
- **A Hippocampus for Linear Attention** — `arXiv:2607.02303` — Read it because it is the paper closest to a Mnemosyne-shaped contribution: a delta-rule compressive state *plus* a bounded exact KV cache, framed as complementary learning systems. Single-author and unreplicated; read it as a design, not a result.
- **Gated DeltaNet-2** — `arXiv:2605.22791` — Read it if you plan to ablate GDN, because it is now the fairer baseline: channel-wise erase and write gates, reported under matched parameter *and* matched state size.
- **RWKV-7 "Goose"** — `arXiv:2503.14456` — Read it for the non-Mamba lineage and the state-tracking-beyond-TC⁰ expressivity argument.
- **xLSTM** — `arXiv:2405.04517` — Read it for the matrix-memory line from a different research tradition, which arrives at similar answers by another route.
- **Variational Linear Attention** — `arXiv:2605.11196` — Read it for the stability framing of associative memory, which is where the interference-bound story is heading.

### hybrid-attention-and-ratios

**must**
- **A Systematic Analysis of Hybrid Linear Attention** — `arXiv:2507.06457` — Read it for the actual ratio evidence: 72 trained models across six linear variants and five ratios, landing on 3:1–6:1 and showing recall collapse as full-attention layers thin. This is the "ratio sets a ceiling" side of the dispute.
- **Hybrid Architectures for Language Models: Systematic Analysis and Design Insights** — `arXiv:2510.04800` — Read it for the two-axis map of the design space: inter-layer (Jamba-style sequential) versus intra-layer (Hymba-style parallel) fusion, scored on long context, scaling, and train/inference cost.
- **Kimi Linear** — `arXiv:2510.26692` — Read it for the current reference point a hybrid has to beat: 3:1 KDA-to-MLA, 75% KV reduction, 6× decode throughput at 1M, and the first fair-comparison claim of a hybrid beating full attention.

**should**
- **Jamba** — `arXiv:2403.19887` — Read it for KV economics stated in systems terms — 4GB at 256K versus 32GB for Mixtral and 128GB for Llama-2-70B — which is the argument that made hybrids a budget question rather than a quality question.
- **Samba** — `arXiv:2406.07522` — Read it for the cleanest statement of the division of labour Laguna relies on: recurrent state compresses history, sliding-window attention holds precise recent memory. Then read `architecture/samba/lit_gpt/model.py:323` and notice the shipped ratio is 1:1, not the 1:6 the writeups condition you to expect.
- **Nemotron-H** — `arXiv:2504.03624` — Read it for how a ratio survives contact with production scale, including pruning and distillation of a hybrid.
- **Rethinking the Role of Efficient Attention in Hybrid Architectures** — `arXiv:2606.15378` — Read it for "Large-Window Laziness" and the opposite conclusion to 2507.06457: the efficient-attention choice governs how *fast* long-context ability emerges rather than its ceiling. Same year, same question, incompatible framings. `research/synthesis.md` leaves this contested and so should you.

**could**
- **Hymba** — `arXiv:2411.13676` — Read it for the parallel intra-layer alternative, where attention and SSM heads run side by side in one layer.
- **Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing** — `arXiv:2607.07953` — Read it for a unified-notation comparison of DeltaNet, GDN, KDA and GDN-2, with hybrid-versus-pure runs at 350M/1.3B/3B — which is our scale, so the numbers transfer.
- **Olmo Hybrid: From Theory to Practice and Back** — `arXiv:2604.03444` — Read it as the most reproducible baseline available: the first fully open 7B hybrid. It prescribes no ratio methodology, which is itself informative.
- **Mechanistic Design and Scaling of Hybrid Architectures** — `arXiv:2403.17844` — Read it for the methodology lineage: 500+ models, synthetic-task-driven topology search. This is where systematic hybrid search started.
- **The MiniMax-M2 Series** — `arXiv:2605.26494` — Read it for the shipping-product retrospective that went the *other* way: full attention on reliability grounds, with hybrids reported fine on benchmarks and deficient on multi-hop reasoning. Commercial incentives attached on both sides; treat the disagreement as the open question of the track.
- **MiniMax-01: Scaling Foundation Models with Lightning Attention** — `arXiv:2501.08313` — Read it for the 456B hybrid that the M2 report walks back, so you can see what was actually abandoned.
- **Short window attention enables long-term memorization** — `arXiv:2509.24552` — Read it for the counterintuitive interaction between window size and what gets memorised, which is directly relevant to Laguna's `[M]` 512-token window.

### long-context-and-effective-context

**must**
- **Lost in the Middle** — `arXiv:2307.03172` — Read it for the U-shaped position bias. For a memory subsystem this is the direct justification for position-aware eviction: cache entries are not equally reachable, so a uniform-value policy is already wrong before you write code.
- **RULER** — `arXiv:2404.06654` — Read it for the harness design you will copy: 13 synthetic tasks across retrieval, multi-hop tracing, aggregation and QA, showing near-perfect NIAH scores collapsing far below claimed length. Read it as methodology, not as a current measurement — and read `measuring-recall-and-memory.md` §1 before you trust the shipped implementation.

**should**
- **Positional Biases Shift as Inputs Approach Context Window Limits** — `arXiv:2508.07479` — Read it for the correction to the naive U-shape story: it holds only to ~50% occupancy, after which primacy decays and the bias becomes distance-based. This changes eviction design, because the position prior you exploit depends on how full the window is.
- **LongBench Pro** — `arXiv:2601.02872` — Read it for the 2026-era number on the effective-versus-advertised gap: 46 models, 8k–256k, naturally occurring tasks. Cite this rather than RULER when you need a current figure.
- **A Structural Theory of Position Bias in Transformers** — `arXiv:2602.16837` — Read it for the mechanistic derivation of the U-shape from causal masking plus residual connections. This is the attribution story rather than the outcome story, which is what this lab is for.
- **Found in the Middle: Calibrating Positional Attention Bias** — `arXiv:2406.16008` — Read it for the opposing claim: up to 15 points recovered by correcting attention bias at inference, implying the bias is substantially correctable rather than architectural. Hold this against 2602.16837 and let the curriculum assert neither.

**could**
- **How to Train Long-Context Language Models (Effectively)** — `arXiv:2410.02660` — Read it for the extension recipe with ablations, and for the evaluation stance: they explicitly reject perplexity and bare NIAH as progress signals. That is the standard our own ablations have to meet.
- **ATLAS: All-round Testing of Long-context Abilities across Scales** — `arXiv:2605.28079` — Read it for rank instability out to 1M: seven models shifting two or more ranks between length regimes, which is the concrete argument against a single headline long-context score.
- **Retrieval Head Mechanistically Explains Long-Context Factuality** — `arXiv:2404.15574` — Read it because retrieval-head masking is one of the six faults in the calibration battery, and this is where the mechanism is characterised.
- **Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings** — `arXiv:2512.12167` — Read it for the NoPE-at-extension result, the cheapest available counterpoint to the YaRN line.
- **Jet-Long: Dynamic Bifocal RoPE** — `arXiv:2607.07740` — Read it for where YaRN's line of work has gone: tuning-free, length-adaptive rescaling paired with a RoPE-faithful local window. Too new to anchor on.
- **Understanding Axes of Difficulty For Long Context Tasks** — `arXiv:2607.08284` — Read it for the decomposition of what makes a long-context task hard, which is what you need before you can claim a harness measures anything.
- **Positional Failures in Long-Context LLMs: A Blind Spot in Reasoning Benchmarks** — `arXiv:2605.23170` — Read it for the crossover: position bias contaminating benchmarks that were not built to measure it.

### agent-memory-in-practice

**must**
- **MemGPT** — `arXiv:2310.08560` — Read it for the original virtual-memory analogy applied to context windows: main versus external context, self-editing memory via tool calls, page-fault-style interrupts. Every later paper assumes this vocabulary. It maps 1:1 onto your intuition, which is exactly why the module spends its length on where it does not.
- **A Survey on Long-Term Memory Security in LLM Agents** — `arXiv:2604.16548` — Read it for the six-phase lifecycle threat model, which shows poisoning as a cross-phase chain (write → persist → propagate → resist-cleanup) rather than a single injection. The right frame the moment Mnemosyne accepts an untrusted write. Cite the v2 title; v1 was titled differently.

**should**
- **A-MEM: Agentic Memory for LLM Agents** — `arXiv:2502.12110` — Read it as the alternative to MemGPT's fixed tiers: Zettelkasten notes, dynamic linking, retroactive refinement on write. Fixed hierarchy versus self-organising index is the core design axis, and you have opinions about it already.
- **AgentPoison** — `arXiv:2407.12784` — Read it for the concrete exploit: triggers optimised so poisoned entries occupy a distinct embedding region, >80% success at <0.1% poison rate with <1% benign degradation. Grounds the surveys in an actual retrieval-level attack.
- **From Untrusted Input to Trusted Memory** — `arXiv:2606.04329` — Read it for the taxonomy of memory *write channels*, including compaction-driven writes — which means your summariser is an attack surface. This is the paper that tells you which code path to instrument.

**could**
- **Memory for Autonomous LLM Agents** — `arXiv:2603.07670` — Read it for the write-manage-read loop and a three-axis taxonomy covering 2022–2026 including benchmarks. The best recent general survey; anything citing only 2024 surveys is stale.
- **From Storage to Experience** — `arXiv:2605.06716` — Read it for the storage → reflection → experience staging, i.e. the clearest statement of why raw logging is not memory.
- **Remembering More, Risking More** — `arXiv:2605.17830` — Read it for the measurement that memory-induced violation rates rise monotonically with exposure length. Any memory eval run over a short horizon systematically under-reports harm — which is a statement about *your* eval design, not theirs.
- **Parallel Context Compaction for Long-Horizon LLM Agent Serving** — `arXiv:2605.23296` — Read it for compaction treated as a serving problem: sequential summarisation blocks inference, so overlap it. Written in your native language of latency, blocking and throughput.
- **Mem0** — `arXiv:2504.19413` — Read it for the production-deployment angle, which the academic surveys under-serve.
- **Control-Plane Placement Shapes Forgetting** — `arXiv:2606.15903` — Read it for the finding that *where* the LLM sits in the memory pipeline determines which failure modes are addressable at all, with mutation-time placement winning. This contradicts the common assumption that retrieval-time reranking is where the leverage is, and it is directly relevant if Mnemosyne has to choose a plug point.

### memory-failure-modes

**must**
- **Alignment Collapse Under KV Cache Quantization** — `arXiv:2606.09864` — Read it for the cleanest single example of the lab's thesis: `[C]` refusals down 15.2% at 1.03× perplexity across 11 models and 1,894 prompts. The outcome metric held; the mechanism broke. If perplexity-only evaluation cannot see a 15-point refusal collapse, it cannot see anything you care about.

**should**
- **SCBench** — `arXiv:2412.10319` — Read it for the result that single-turn rankings do not survive multi-turn cache reuse — i.e. the standard evaluation regime is not the deployment regime, and the ordering changes when you fix that.

**could**
- **KVSink** — `arXiv:2508.04257` — Read it for attention-sink destruction as the named mechanism behind 2-bit KV failure, one of the two mechanisms offered in the sub-4-bit dispute.
- **KVarN** — `arXiv:2606.03458` — Read it for the other one: error accumulation over long reasoning chains, which is why 2-bit is perplexity-friendly and reasoning-hostile.
- **MemDelta: Controlled Baselines and Hidden Confounds in Agent Memory Evaluation** — `arXiv:2606.29914` — Read it for the confound catalogue. It is the closest thing to a pre-mortem for the harness we are about to build.

### measuring-memory

**must**
- Re-read **From Tensor Buffer to Distributed Memory Hierarchy** — `arXiv:2607.02574` — this time for the seven measurement gaps only. Metadata-to-data ratio, handoff granularity cost, prefix hit rate versus lookup overhead, cross-tier migration latency under eviction, ownership coordination overhead, multi-tenant isolation semantics, durability contracts for persisted KV. `[A]` Two of those seven are measurable on a single machine and are the cheapest publishable methodology available to this lab.

**should**
- **HELMET** — `arXiv:2410.02694` — Read it for how a long-context benchmark suite should be constructed and reported, and for its explicit treatment of what its own numbers do not support.
- **NoLiMa: Long-Context Evaluation Beyond Literal Matching** — `arXiv:2502.05167` — Read it for the removal of lexical overlap between needle and question. This is the fix for the defect `measuring-recall-and-memory.md` finds in this repo's own RULER implementation, where the question template repeats the needle's own words.
- **Quantifying Variance in Evaluation Benchmarks** — `arXiv:2406.10229` — Read it for seed-to-seed and monotonicity variance measurements. You cannot claim a policy helped until you know your null, and this is where the null comes from.

**could**
- **Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions** — `arXiv:2507.05257` — Read it for the benchmark design that makes memory rather than retrieval the thing under test.

---

## Track D — Training systems

Four modules: `distributed-training-strategies`, `checkpointing-and-resumption`,
`determinism-and-reproducibility`, `training-telemetry-as-observability`.

**Read this tier differently.** `[M]` `torch.distributed.is_available()` is **False** on
the lab wheel and `torch._C._distributed_c10d` does not exist in the build
(`ASSUMPTIONS.md → single-device-only`). Nothing in the parallelism half runs here. Read
it as design literature and as vocabulary for renting hardware later — which, given your
background, will be the fastest track on the list and the one where you should be
sceptical of the module rather than the reverse.

### must

- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** — `arXiv:1910.02054` — *with* `distributed-training-strategies`. Read it for the three-stage decomposition (optimizer state, gradients, parameters) and the memory arithmetic per stage. This is sharded replication with a communication bill, and the bill is itemised.
- **PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel** — `arXiv:2304.11277` — Read it for what ZeRO looks like when it has to survive an actual framework: the flat-parameter unit, prefetch overlap, and the failure modes that only show up in implementation.
- **Megatron-LM** — `arXiv:1909.08053` — Read it for tensor parallelism as it was originally stated: split the weight matrices, insert two all-reduces per block, done. Nine pages, and it is the paper that makes "TP is bandwidth-expensive and therefore intra-node" obvious rather than memorised.

### should

- **PyTorch Distributed: Experiences on Accelerating Data Parallel Training** — `arXiv:2006.15704` — Read it for gradient bucketing and the overlap of backward with all-reduce — the write-combining idea you will recognise immediately, and the origin of `no_sync()`.
- **Efficient Large-Scale Language Model Training on GPU Clusters** — `arXiv:2104.04473` — Read it for 3D parallelism composed and costed: which axis to spend where, with a model that predicts the answer.
- **GPipe** — `arXiv:1811.06965` — Read it for the pipeline bubble stated as a fraction, and the microbatch count that shrinks it. The clearest possible statement of the pipeline tradeoff.
- **Zero Bubble Pipeline Parallelism** — `arXiv:2401.10241` — Read it for the observation that backward is two separable operations (input grad, weight grad) and scheduling them independently removes the bubble. Contested once you count activation memory.
- **GShard** — `arXiv:2006.16668` — Read it for expert parallelism and the all-to-all that defines it, which is the communication pattern MoE training lives or dies on.
- **Reducing Activation Recomputation in Large Transformer Models** — `arXiv:2205.05198` — Read it for selective recomputation and sequence parallelism, and for the activation-memory formula you will actually use when sizing a run.
- **MegaScale** — `arXiv:2402.15627` — Read it as a site-reliability document: most of the paper is straggler detection and eviction, which is the part nobody writes about and the part you already know how to reason about.
- **The Llama 3 Herd of Models** — `arXiv:2407.21783` — Read the infrastructure section for documented failure rates at scale. It is the best public evidence that training at scale is an availability problem wearing an ML costume.

### could

- **Alpa: Automating Inter- and Intra-Operator Parallelism** — `arXiv:2201.12023` — Read it for the framing of plan selection as a search problem, which is still not solved and is the honest frontier of this module.
- **Streaming DiLoCo with overlapping communication** — `arXiv:2501.18512` — Read it for the low-communication branch: infrequent outer synchronisation, which is the only line of work that would ever make multi-machine training viable on hardware like ours.
- **Universal Checkpointing** — `arXiv:2406.18820` — *with* `checkpointing-and-resumption`. Read it for the format that decouples checkpoint layout from the parallelism topology that wrote it. This is the DR-portability problem stated properly.
- **DataStates-LLM: Lazy Asynchronous Checkpointing** — `arXiv:2406.10707` — Read it for the staging pipeline that removes checkpointing from the critical path, and for the consistency argument that makes it safe.
- **Understanding LLM Checkpoint/Restore I/O Strategies and Patterns** — `arXiv:2512.24511` — Read it for actual I/O characterisation — sizes, rates, patterns — which is what you need to size a checkpoint budget rather than guess one.
- **TierCheck: Tiered Checkpointing for Fault Tolerance** — `arXiv:2605.17821` — Read it for checkpointing as a storage-tier problem, which is the framing your instincts will reach for anyway.
- **BitSnap: Checkpoint Sparsification and Quantization** — `arXiv:2511.12376` — Read it for the checkpoint-compression tradeoff, and for what it costs in restore fidelity.
- **All is Not Lost: LLM Recovery without Checkpoints** — `arXiv:2506.15461` — Read it for the alternative position: reconstruct rather than restore. Whether that is a real option at our scale is an open question.
- **Why Atomicity Matters to AI/ML Infrastructure** — `arXiv:2603.02603` — Read it for the atomicity argument stated in storage terms, which pairs directly with `training/olmo-core/src/olmo_core/train/checkpoint.py:498` and its directory-rename commit protocol.
- **Exploring Silent Data Corruption as a Reliability Challenge in LLM Training** — `arXiv:2604.00726` — Read it because silent corruption is the failure mode with no alarm, and this module's whole point is that training has several of those.
- **Robust LLM Training Infrastructure at ByteDance** — `arXiv:2509.16293` — Read it for a second production reliability account to triangulate against MegaScale and Llama 3.
- **Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference** — `arXiv:2506.09501` — *with* `determinism-and-reproducibility`. Read it for the batch-size-dependence result: the same prompt gives different logits depending on what else is in the batch. Determinism is not a seed property.
- **Deterministic Inference across Tensor Parallel Sizes** — `arXiv:2511.17826` — Read it for the reduction-order argument, i.e. why changing your topology changes your numbers.
- **LLM-42: Enabling Determinism in LLM Inference with Verified Speculation** — `arXiv:2601.17768` — Read it for a constructive fix rather than a diagnosis.
- **Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping** — `arXiv:2002.06305` — Read it for the seed-variance measurement that predates the current interest, and because its magnitudes are still the right order.
- **Assessing the Macro and Micro Effects of Random Seeds on Fine-Tuning LLMs** — `arXiv:2503.07329` — Read it for the modern replication, and for how many seeds you actually need before a difference means anything.
- **Reproducibility is the New Copyleft** — `arXiv:2606.03019` — Read it for the reproducible-builds framing, which is a much better model for a research lab than "we set a seed."

---

## Track E — Post-training and evaluation

Three modules: `supervised-and-preference-finetuning`, `building-an-eval-you-can-trust`,
`measuring-recall-and-memory`.

**Read this tier sceptically too.** The post-training module argues against running most of
what it teaches, and it is right: `[C]` an auditable single-GPU study at 135M found GSM8K
exact match *fell* under RLVR (`arXiv:2606.22189`). Read for vocabulary and for the eval
half, which is where this lab's contribution actually lives.

### must

- **Training language models to follow instructions with human feedback (InstructGPT)** — `arXiv:2203.02155` — Read it for the three-stage pipeline stated once, cleanly, before the field fragmented. Everything after is a substitution into one of those three stages.
- **Direct Preference Optimization** — `arXiv:2305.18290` — Read it for the derivation that removes the reward model — the implicit-reward reparameterisation — and note the memory consequence: you hold a reference-model forward pass, which is a KV and activation cost the RLHF version hid inside a separate service.
- **DeepSeekMath (GRPO)** — `arXiv:2402.03300` — Read it for GRPO: drop the value network, normalise advantages within a sampled group. That is the algorithm every 2025–2026 reasoning run is built on, stated in two paragraphs.
- **Quantifying Variance in Evaluation Benchmarks** — `arXiv:2406.10229` — *with* `building-an-eval-you-can-trust`. Listed in Track C too, and it earns both. Read it for the noise floor. Without one you cannot distinguish a result from a seed.

### should

- **Learning to summarize from human feedback** — `arXiv:2009.01325` — Read it for the reward-model-overoptimisation curve, which is the clearest picture of why a learned reward is a proxy and proxies get gamed.
- **Proximal Policy Optimization Algorithms** — `arXiv:1707.06347` — Read the clipped objective and stop. You need it only to understand what GRPO removed.
- **DeepSeek-R1** — `arXiv:2501.12948` — Read it for RLVR at scale with a verifiable reward and no reward model, and for the cold-start SFT stage they had to add back.
- **Tulu 3** — `arXiv:2411.15124` — Read it as the fully-open post-training recipe with data and evaluation released, which is the only one you could actually reproduce.
- **LIMA: Less Is More for Alignment** — `arXiv:2305.11206` — Read it for the 1,000-example result and its stated limits. At our scale this is the difference between a post-training experiment being affordable and being fantasy.
- **Does Reinforcement Learning Really Incentivize Reasoning Capacity Beyond the Base Model?** — `arXiv:2504.13837` — Read it for the pass@k analysis suggesting RLVR sharpens the base distribution rather than extending it. This is the load-bearing objection to the whole RLVR programme.
- **Are Emergent Abilities of Large Language Models a Mirage?** — `arXiv:2304.15004` — *with* `building-an-eval-you-can-trust`. Read it for the argument that the metric, not the model, produces the discontinuity. Then apply it to your own memory metrics, where the same trap is wide open.
- **HELMET** — `arXiv:2410.02694` — see Track C.
- **Rethinking Benchmark and Contamination for Language Models with Rephrased Samples** — `arXiv:2311.04850` — Read it for the demonstration that n-gram contamination checks are trivially evaded by paraphrase.

### could

- **Understanding R1-Zero-Like Training: A Critical Perspective** — `arXiv:2503.20783` — Read it for the finding that some reported RLVR gains are template and normalisation artifacts.
- **A Sober Look at Progress in Language Model Reasoning** — `arXiv:2504.07086` — Read it for the reproducibility audit: how much of the reported reasoning progress survives careful re-evaluation.
- **L20-Edu-135M: An Auditable Single-GPU Study** — `arXiv:2606.22189` — Read it because it ran our experiment at our scale on one GPU and reported a *regression*. `research/synthesis.md` cites this as a reason to park RLVR; read the primary source before accepting the park.
- **GSM-Symbolic** — `arXiv:2410.05229` — Read it for the template-perturbation method, which is the cheapest available contamination probe and directly reusable for memory tasks.
- **LiveCodeBench** — `arXiv:2403.07974` — Read it for the time-windowed construction that makes contamination measurable rather than assumed.
- **LiveBench** — `arXiv:2406.19314` — Read it for the same idea applied broadly, with objective ground truth.
- **A Survey on Data Contamination for Large Language Models** — `arXiv:2502.14425` — Read it for the taxonomy of contamination types when you need to say precisely which one you controlled for.
- **A Comprehensive Survey of Contamination Detection Methods** — `arXiv:2404.00699` — Read it for the detection side, including what each method cannot see.
- **Evading Data Contamination Detection for Language Models is (too) Easy** — `arXiv:2402.02823` — Read it for the adversarial view, which is the right prior when reading anyone's contamination claim.
- **NoLiMa** — `arXiv:2502.05167` — see Track C.
- **Retrieval Or Holistic Understanding? Dolce** — `arXiv:2409.06338` — Read it for the decomposition of a long-context task into retrieval and holistic components, which is the axis `measuring-recall-and-memory.md` builds its task suite on.
- **100-LongBench** — `arXiv:2505.19293` — Read it for the audit of whether de facto long-context benchmarks measure length at all.

---

## Track F — Inference

Three modules: `quantization`, `speculative-decoding-and-serving`, `running-laguna-locally`.

On this machine you get the *capacity* half of quantization and not the *arithmetic*
half — `torch._scaled_mm` refuses on gfx1151 and no Marlin/CUTLASS low-bit path exists on
ROCm. That is tagged `[M]` in `quantization.md` §1 with a fresh-process reproduction, but
it has **no row in `ASSUMPTIONS.md`**, so treat it as a module-level measurement pending
registration rather than as a register entry. Read the weight-quantization papers for the
mechanism; expect to measure only bytes.

### must

- **Fast Inference from Transformers via Speculative Decoding** — `arXiv:2211.17192` — Read it for the acceptance-sampling proof that the output distribution is *exactly* preserved. That exactness is the entire reason speculation is allowed in a serving path, and it is the part most explanations garble.
- **GPTQ** — `arXiv:2210.17323` — Read it for one-shot layerwise weight quantization with second-order error compensation, i.e. the first method that made 4-bit weights routine.
- **AWQ** — `arXiv:2306.00978` — Read it for the activation-aware insight: a small fraction of weight channels matter disproportionately, and scaling beats keeping them in fp16. Also the format behind most quantized checkpoints you will actually download.

### should

- **Accelerating Large Language Model Decoding with Speculative Sampling** — `arXiv:2302.01318` — Read it as the independent contemporaneous derivation, and for the batching interaction that decides whether speculation pays at all.
- **EAGLE** — `arXiv:2401.15077` — Read it for the move from token-space to feature-space drafting, which is the version that actually shipped.
- **LLM.int8()** — `arXiv:2208.07339` — Read it for the outlier-feature discovery. This is where the field learned that transformers have a small number of dimensions that refuse to be quantized, and it is the same fact that reappears in KIVI's per-channel keys.
- **SmoothQuant** — `arXiv:2211.10438` — Read it for the migration trick: shift quantization difficulty from activations to weights with a per-channel scale. The cleanest example of "change the tensor before you pick the scale."
- **FP8 Formats for Deep Learning** — `arXiv:2209.05433` — Read it for E4M3 versus E5M2 and why you need both. Also read it against `[M]` our own finding that hipBLASLt configuration moves length-1M bf16 reduction error by **2.8×** (2.01e-3 versus 5.60e-3) — the format is not the only thing determining your error budget (`ASSUMPTIONS.md → hipblaslt-config`).
- **Sarathi-Serve** — `arXiv:2403.02310` — see Track C.
- **KVQuant** — `arXiv:2401.18079` — Read it for the four mechanisms that get below 4 bits — pre-RoPE key quantization, per-channel keys, sensitivity-weighted datatypes, dense-and-sparse outlier isolation — and for a worked account of where the error comes from.
- **KIVI** — `arXiv:2402.02750` — Read it for the empirical asymmetry every later KV quantizer inherits: keys per-channel, values per-token — and for the layout that keeps it kernel-friendly during streaming decode.

### could

- **Medusa** — `arXiv:2401.10774` — Read it for the multiple-heads alternative to a draft model, which removes the second model entirely.
- **EAGLE-2** — `arXiv:2406.16858` — Read it for dynamic draft trees, i.e. spending the draft budget where acceptance is likely.
- **Unlocking Efficiency in LLM Inference: A Survey of Speculative Decoding** — `arXiv:2401.07851` — Read it when you need the design-space map rather than one method.
- **DFlash: Block Diffusion for Flash Speculative Decoding** — `arXiv:2602.06036` — Read it because it is the speculative-decoding scheme in the Laguna llama.cpp branch we cloned (`PROVENANCE.md`), so it is the one you can actually read source for.
- **Quantize the Target, Quantize the Drafter** — `arXiv:2607.04244` — Read it for the composition question nobody usually asks: what happens when both models in a speculative pair are quantized.
- **When Quantization Is Free: An int4 KV Cache That Outruns fp16 on Apple Silicon** — `arXiv:2605.05699` — Read it because it is the closest published analogue to our platform: unified memory, bandwidth-bound decode, and a KV format that wins on bytes rather than on math.
- **Alignment Collapse Under KV Cache Quantization** — `arXiv:2606.09864` — see Track C, and read it again here as a quantization result rather than a memory result.
- **KVSink** — `arXiv:2508.04257` — see Track C.
- **KVarN** — `arXiv:2606.03458` — see Track C.

---

## Contested pairs — read both, take neither side

The house rule is that contested topics are presented as contested. These are the five
live disputes where reading only one side will leave you confidently wrong. `[C]` all
positions are from the papers named.

| Dispute | One side | The other |
|---|---|---|
| Does the hybrid ratio set a capability ceiling, or only the rate at which long-context ability emerges? | `arXiv:2507.06457` | `arXiv:2606.15378` |
| Is efficient attention worth it at production scale? | `arXiv:2510.26692` (Kimi Linear: yes, under matched pretraining) | `arXiv:2605.26494` (MiniMax M2: no, on reliability grounds) |
| Paging versus contiguous virtual memory for KV. | `arXiv:2309.06180` | `arXiv:2405.04437` |
| Is position bias architectural or correctable at inference? | `arXiv:2602.16837` | `arXiv:2406.16008` |
| Are attention sinks necessary, or an artifact to be removed? | `arXiv:2603.11487` | `arXiv:2504.20966` |

Two more the anchors flag and this list inherits: whether sub-4-bit KV is deployable
(`arXiv:2402.02750` and `arXiv:2401.18079` claim near-lossless 2-bit; `arXiv:2508.04257`
and `arXiv:2606.03458` name the mechanisms by which it fails on reasoning), and whether
the KV cache is "memory" at all — the serving literature (`arXiv:2412.19442`,
`arXiv:2607.02574`) says tensor buffer, the agent-memory literature (`arXiv:2505.00675`,
`arXiv:2507.03724`) says first-class tier, and the two communities rarely cross-cite.
`research/synthesis.md` takes the sharper position that it is "a memo table with no
backing store: no fault path, no miss signal, no durability contract" — read that against
both camps.

---

## What is deliberately not here

- **Anything I could not resolve to a real arXiv id.** Four items are named in
  `research/reference/papers/README.md` as excluded for exactly this reason — a
  representation×lifecycle agent-memory grid on OpenReview, Chroma's "Context Rot" tech
  report, TurboQuant and PolarQuant. They may be worth fetching directly; they are not
  worth a fabricated identifier.
- **A Zamba2 paper.** No reliable id was established. Read the code instead —
  `PROVENANCE.md` has the clone at `1b182f40f225`, and `CODE_MAP.md`'s Samba section
  describes its shared-attention-block design by contrast.
- **Textbooks and courses.** You learn by building and by reading source; a textbook is a
  slower path to both.
- **The 132 arXiv ids in `curriculum/citation-verification.json` marked unreachable.** They
  are unchecked, not bad. Nothing here depends on one.

---

## Honest weaknesses of this list

`[A]` Four, stated so you can discount appropriately.

1. **The must tier is too big to be a queue.** 37 papers is six weeks of your actual
   budget. Treat the five-item opener as the queue and the rest as a table.
2. **The 2026 entries are ranked on abstract and structure, not on replication.** Roughly a
   quarter of this list is from the last six months, which is deliberate — a 2024 reading
   list for KV memory is now misleading — but recency and reliability trade against each
   other and I have taken the recency side. Anything dated 26xx should be read as a
   proposal.
3. **Track A is over-served relative to its value.** The foundations papers are here for
   completeness. If you skip them entirely and read nanoGPT plus the modules, you will lose
   very little.
4. **Ranking is one person's judgement.** The `must`/`should`/`could` split is `[A]`, not
   measured. Where a rank differs from `research/reference/papers/README.md` I have said so
   in-line (there is exactly one: `arXiv:1911.02150` promoted from *should* to first read).

---

## Decision / Riskiest assumption / Next test

**Decision.** Read the five, then `research/reference/CODE_MAP.md` end to end, then
`models/laguna-s/config.json` with a notebook open. Then start Track C's must tier and pull
from the other tracks only when a module points you at one. Do not read this list linearly.

**Riskiest assumption.** That the ranking transfers — that a paper I marked *must* is
actually the one that unblocks *you*, given a background none of these authors wrote for.
The failure mode is silent: you read the must tier, learn a great deal, and still cannot
design the oracle-diff harness, because the thing that would have unblocked you was ranked
*could*.

**Next test.** After the first two Track C modules, list the three papers you actually
opened and the three you wished you had. If the overlap with the must tier is under half,
the ranking is wrong and this file gets a `version: 1.1.0` rather than an apology.
