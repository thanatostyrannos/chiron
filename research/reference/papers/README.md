# papers/ — the anchoring reading list

**76 papers**, every one resolved against the live arXiv API. 24 must-read, 36 should-read, 16 could-read.

`anchors.bib` is generated, not hand-written. Titles, authors and dates come from the API rather than from anyone's memory, because a citation that looks right and is wrong is invisible to a reader and propagates into everything that cites it. Regenerate with:

```
python scripts/verify_papers.py <candidates.json> --out research/reference/papers
python scripts/generate_papers.py research/reference/papers/resolved_papers.json \
    --recency <recency_notes.json>
```

The verifier rejects a claimed arXiv id that resolves to a *different* paper, and distinguishes 'unreachable' from 'does not exist' — a network timeout is not evidence about a paper. PDFs are not committed; fetch what you need from the `url` field.

## Coverage by year

| 2019 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| 1 | 1 | 3 | 12 | 24 | 15 | 20 |

20 of 76 are from 2026 — this list was built against current arXiv, not from training data.

## Memory taxonomy — fixing the vocabulary

**must**

- [Rethinking Memory in LLM based Agents: Representations, Operations, and Emerging Topics](https://arxiv.org/abs/2505.00675v3) (2025, `2505.00675v3`) — Read this for the cleanest split of memory into parametric vs contextual (structured/unstructured) plus six atomic operations - consolidation, updating, indexing, forgetting, retrieval, compression - an operation vocabulary that maps almost 1:1 onto cache and storage primitives.
- [Memory for Autonomous LLM Agents:Mechanisms, Evaluation, and Emerging Frontiers](https://arxiv.org/abs/2603.07670v1) (2026, `2603.07670v1`) — Read this for the best recent (Mar 2026) agent-memory survey: it formalizes agent memory as a write-manage-read loop with a three-axis taxonomy (temporal scope, representational substrate, control policy) and covers 2022-2026, including evaluation benchmarks.
- [What to Keep, What to Forget: A Rate--Distortion View of Memory Compaction in LLMs and Agents](https://arxiv.org/abs/2607.08032v1) (2026, `2607.08032v1`) — Read this for the argument that KV-cache eviction, prompt pruning, recurrent state bounding, and agent memory consolidation are one problem under a resource budget — a seven-axis taxonomy that lets you talk about all four legs in a single vocabulary, plus the recurring failure mode (attention/recency signals discard irreversibly before the query is known).

**should**

- [A Survey on Large Language Model Acceleration based on KV Cache Management](https://arxiv.org/abs/2412.19442v3) (2024, `2412.19442v3`) — Read this for the canonical token-level / model-level / system-level partition of KV cache work (selection, budget allocation, merging, quantization, low-rank decomposition) — the reference map most later KV papers position themselves against.
- [From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs](https://arxiv.org/abs/2504.15965v2) (2025, `2504.15965v2`) — Read this for the object/form/time three-dimensional framework that yields eight memory categories, and for an explicit mapping between human-memory terms and their LLM counterparts — useful precisely because it shows where the cognitive analogy stops paying.
- [MemOS: A Memory OS for AI System](https://arxiv.org/abs/2507.03724v4) (2025, `2507.03724v4`) — Read this for the parametric / activation / plaintext trichotomy and the OS framing (MemCube units with provenance and versioning, a scheduler that migrates content between tiers) — the closest thing in the literature to a storage-hierarchy mental model for LLM memory.
- [From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving](https://arxiv.org/abs/2607.02574v1) (2026, `2607.02574v1`) — Read this for the systems-side vocabulary of KV cache: locality, lifetime, ownership, substrate, and the five recurring architectures (local-paged, disaggregated-pipeline, shared-store, memory-pool, hybrid-tier), plus seven named gaps in how KV systems are evaluated.

**could**

- [A Survey on the Memory Mechanism of Large Language Model based Agents](https://arxiv.org/abs/2404.13501v1) (2024, `2404.13501v1`) — Read this for provenance: the origin-point agent-memory survey whose design/evaluation split most later taxonomies inherit, so you can see which distinctions are load-bearing and which are just carried forward.
- [From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms](https://arxiv.org/abs/2605.06716v1) (2026, `2605.06716v1`) — Read this for the storage -> reflection -> experience staging of agent memory (trajectory preservation, refinement, abstraction) — the clearest recent statement of why raw logging is not memory and what proactive exploration and cross-trajectory abstraction add.

> **Recency and contested points.** The last six months (Feb-Jul 2026) produced a distinct wave that a pre-2026 reading list misses entirely, and three of the nine entries above come from it. 2607.08032 (Jul 2026) is the newest and the most structurally useful: it is the first paper to argue that KV-cache eviction, prompt compression, architectural/recurrent state bounding, and agent memory consolidation are the same rate-distortion problem, which is exactly the unification this track needs. 2607.02574 (Jun/Jul 2026) is the current state of the serving-side KV vocabulary (locality/lifetime/ownership/substrate) and supersedes 2412.19442 for distributed and tiered deployments while not replacing it for algorithm-level technique coverage. 2603.07670 (Mar 2026) is the best recent general agent-memory survey and covers benchmarks through early 2026. 2605.06716 (May 2026) adds the experience-abstraction framing.

CONTESTED, and worth presenting as contested rather than resolved: (1) Whether the KV cache is "memory" at all. The serving literature (2412.19442, 2607.02574) treats it as a tensor buffer / storage object with a lifetime and an owner; the agent-memory literature (2505.00675, 2507.03724) treats it as a first-class memory tier ("activation memory" / "intermediate latent"). The two communities use incompatible vocabulary for the same bytes and cross-cite rarely. 2607.08032 is the first serious attempt to bridge them and is too new to have settled anything. (2) Whether human-memory analogies (episodic / semantic / procedural, short-term vs long-term) earn their keep. 2504.15965 and 2605.06716 lean on them structurally; 2603.07670 and 2607.08032 deliberately use functional/mechanistic axes instead and implicitly treat the cognitive mapping as decorative. Pick neither side in the curriculum -- note which axis each source is using. (3) Whether "parametric memory" belongs in the same taxonomy as the rest at all, since it is written by gradient descent rather than by a runtime operation; 2505.00675 and 2507.03724 include it and admit the lifecycle mismatch.

One relevant item deliberately EXCLUDED because I could not establish an arXiv id: "LLM Agent Memory: A Survey from a Unified Representation-Management Perspective" (Mar 2026, OpenReview forum KPs1EgGKcT, also Preprints.org 202603.0359). It proposes three paradigms -- natural-language tokens, intermediate representations, parameters -- crossed with three management stages (construction, update, query), which is arguably the tidiest representation x lifecycle grid available. It appears to be OpenReview/Preprints only, so it would fail arXiv id verification; fetch it directly if the grid is wanted.

## KV cache mechanics and reduction

**must**

- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245v3) (2023, `2305.13245v3`) — Read this for the KV-cache sizing arithmetic and the head-sharing knob (H → G → 1) that essentially every shipped open model now sets — plus the uptraining recipe that converts an MHA checkpoint at ~5% of pretrain compute.
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434v5) (2024, `2405.04434v5`) — Read this for Multi-head Latent Attention: the low-rank latent KV projection that compresses per-token cache 93.3% without the quality loss GQA/MQA pay, and for the decoupled-RoPE trick that makes low-rank KV compatible with rotary position.

**should**

- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150v1) (2019, `1911.02150v1`) — Read this for the original, cleanest statement of why autoregressive decode is memory-bandwidth-bound rather than FLOPS-bound — the single mental model a storage/caching engineer needs before anything else in this track makes sense.
- [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://arxiv.org/abs/2401.18079v6) (2024, `2401.18079v6`) — Read this for the four mechanisms that get you below 4 bits — pre-RoPE key quantization, per-channel keys, sensitivity-weighted non-uniform datatypes, and per-vector dense-and-sparse outlier isolation — and for a worked account of where the error actually comes from.
- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750v2) (2024, `2402.02750v2`) — Read this for the empirical asymmetry every later KV quantizer inherits: keys have per-channel outliers so quantize keys per-channel, values do not so quantize per-token — and for the grouped/residual layout that makes it kernel-friendly during streaming decode.
- [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348v1) (2026, `2606.19348v1`) — Read this for the current frontier design point: Compressed Sparse Attention plus Heavily Compressed Attention compress along the *sequence* axis (groups of m tokens → one KV entry), an axis orthogonal to MLA's per-token compression, reported at 10% of V3.2's KV size at 1M context.

**could**

- [Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](https://arxiv.org/abs/2502.14837v2) (2025, `2502.14837v2`) — Read this for the concrete MHA/GQA → MLA conversion recipe (partial-RoPE removal plus joint SVD of K and V), recoverable on 0.3–0.6% of the data — the cheapest way to run an MLA-vs-GQA ablation without pretraining two models.
- [Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization](https://arxiv.org/abs/2607.08057v1) (2026, `2607.08057v1`) — Read this for the freshest (Jul 2026, ACL 2026 Findings) systems-vocabulary framing — temporal (scheduling), spatial (placement/migration/tiering), structural (representation/retention) — which maps KV caching onto storage-hierarchy concepts you already own.

> **Recency and contested points.** Not a stale track — the last ~6 months moved it materially, and a list stopping at KIVI/KVQuant/DeepSeek-V2 would be a 2024 answer.

What is new and load-bearing:
1. Compression has moved from the *feature* axis to the *sequence* axis. DeepSeek-V3.2 (arXiv 2512.02556, Dec 2025) introduced DeepSeek Sparse Attention on top of MLA; DeepSeek-V4 (arXiv 2606.19348, Jun 2026) then added Compressed Sparse Attention and Heavily Compressed Attention, which fold every m tokens into one KV entry. MLA shrinks each token's KV; CSA/HCA shrink how many entries exist. Treat these as two orthogonal knobs — a 2026 KV-budget experiment that only varies one is under-designed. V4 reports V4-Flash at ~7% of V3.2's KV size and ~10% of its per-token FLOPs at 1M context (vendor-reported, not independently replicated).
2. There is a newer survey than the canonical one. arXiv 2607.08057 (Jul 9 2026, ACL 2026 Findings) reorganizes the field around serving-system concerns rather than algorithmic taxonomy. I kept 2412.19442 at 'must' because its taxonomy is still the better first read and it is far more cited; 2607.08057 is the better second read for this specific reader.
3. Safety/alignment degradation under KV quantization is a newly opened front, not present in the 2024 literature: arXiv 2606.09864 (Jun 2026) reports refusal-rate collapse (e.g. ~15% refusal loss on Mistral-7B) at perplexity deltas small enough that standard PPL-only evaluation misses it entirely. Relevant to this lab because it is exactly an attribution failure — outcome metric fine, mechanism broken.

CONTESTED — do not let the list imply consensus:
- **Is sub-4-bit KV actually deployable?** KIVI and KVQuant claim near-lossless 2–3 bit. Multiple 2025–2026 results push back: 4-bit matches FP16 closely but 2-bit degrades substantially on reasoning and generation, with attention-sink destruction (KVSink, arXiv 2508.04257) and error accumulation in long reasoning chains (KVarN, arXiv 2606.03458) offered as mechanisms. Newer entrants (Kitty, arXiv 2511.18643, dynamic channel-wise precision boost) exist precisely because 2-bit is not settled. The honest summary for 2026: 8-bit/FP8 is production-boring, 4-bit is broadly safe, 2-bit is a live research question whose answer depends heavily on task (perplexity-friendly, reasoning-hostile).
- **Cross-layer KV sharing vs within-layer sharing.** CLA/YOCO-style cross-layer sharing keeps producing papers (xKV arXiv 2503.18893, CommonKV arXiv 2508.16134, cross-layer fusion arXiv 2512.03870), but the recurring finding is that cross-layer methods still tend to underperform within-layer GQA at matched budget. Unresolved; do not present cross-layer as a strict improvement.
- **MLA vs GQA at small scale.** MLA's wins are reported at 100B+ MoE scale. Whether the low-rank latent survives at the 20M–300M ablation scale this lab runs is, as far as I can find, untested in public — which makes it a genuinely available experiment rather than a replication.

Two items I saw reported but could NOT pin to an arXiv id, so they are deliberately excluded from the paper list and flagged here instead: TurboQuant (Google Research, reported ICLR 2026) and PolarQuant (reported AISTATS 2026), both rotation-before-quantization KV quantizers. Worth a 10-minute lookup before you finalize a quantization reading order.

## KV compression and eviction

**must**

- [H$_2$O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048v3) (2023, `2306.14048v3`) — Read this for the canonical accumulated-attention eviction policy and its dynamic-submodular framing — the baseline any pluggable policy interface must be able to express.
- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453v4) (2023, `2309.17453v4`) — Read this for the attention-sink result — why evicting the first few tokens collapses the model, and why every eviction policy since pins a prefix.
- [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469v2) (2024, `2404.14469v2`) — Read this for the observation-window trick (compress at prefill using the last ~32 query positions) — the mechanism PyramidKV, Ada-KV, FastKV and RocketKV all extend.

**should**

- [PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling](https://arxiv.org/abs/2406.02069v4) (2024, `2406.02069v4`) — Read this for non-uniform per-layer budget allocation and the depth-wise attention-concentration evidence behind it — plus the caveat that it converges to SnapKV as the compression ratio rises.
- [KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments](https://arxiv.org/abs/2504.15364v4) (2025, `2504.15364v4`) — Read this for eviction that never reads an attention score (key cosine distinctiveness) — the option you need when fused/flash-style kernels never materialise the attention matrix.
- [The Pitfalls of KV Cache Compression](https://arxiv.org/abs/2510.00231v2) (2025, `2510.00231v2`) — Read this for the failure modes LongBench hides: under StreamingLLM/SnapKV/TOVA/H2O/K-Norm some instructions are dropped entirely, with system-prompt leakage as the worked case study. ACL 2026, revised May 2026.
- [KV Cache Optimization Strategies for Scalable and Efficient LLM Inference](https://arxiv.org/abs/2603.20397v1) (2026, `2603.20397v1`) — Read this for the current (Mar 2026) map: five technique families scored against seven deployment scenarios, with the explicit finding that no single method dominates.

**could**

- [RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](https://arxiv.org/abs/2502.14051v3) (2025, `2502.14051v3`) — Read this for the argument that permanent eviction and dynamic sparse attention are complementary, not rivals — coarse eviction first, then fine-grained top-k — with decode-phase memory and bandwidth numbers attached.
- [Learning to Evict from Key-Value Cache](https://arxiv.org/abs/2602.10238v2) (2026, `2602.10238v2`) — Read this for the learned-policy alternative to heuristics: lightweight per-head RL agents ranking tokens by predicted future usefulness — the direct analogue of learned cache replacement.

> **Recency and contested points.** Last ~6 months (Feb-Jul 2026) is where this track actually moved, and a pre-2025 reading list is now misleading.

NEW SINCE FEB 2026 (ids seen in search results this session; only the ones I fetched are marked verified above): (1) Learned rather than heuristic eviction — "Learning to Evict from Key-Value Cache" (2602.10238, v2 Jun 2026) trains per-head RL policies and is the first credible break from attention-mass heuristics. (2) Reasoning-model eviction is the dominant subthread, because policies tuned for long *input* prompts degrade badly on long chain-of-thought *generation*: ForesightKV (2602.03203), LookaheadKV (2603.10899), MomentKV (2606.01563, ICLR 2026), Value-Aware Stochastic KV Cache Eviction / VaSE (2606.03928), KVpop (2607.05061). (3) Diagnostics over leaderboards — "When Does Value-Aware KV Eviction Help? A Fixed-Contract Diagnostic for Non-Monotone Cache Compression" (2605.08234, verified) argues task accuracy alone cannot tell you *why* a selector worked. (4) Two 2026 surveys worth knowing: 2603.20397 (technique-to-scenario mapping, included above) and "From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving" (2607.02574, 30 Jun 2026, verified) — thirty-plus serving systems on four design axes, explicitly naming tiered eviction and shared-cache semantics as open problems. That second one is the closest thing in the literature to a storage-hierarchy framing and is probably the single best bridge document for a systems reader, though it sits on the serving side of this track's boundary.

CONTESTED — do not let the list above imply consensus:
- Eviction vs. retention. Permanent eviction (H2O/SnapKV) cuts peak capacity but is irreversible; full-retention sparse loading (Quest/SparQ family) preserves fidelity and cuts bandwidth but not capacity, usually pushing the cache to CPU/offload tiers. RocketKV (2502.14051) claims they are orthogonal and composable; other 2026 work treats eviction as the wrong primitive entirely and prefers tiered offload plus retrieval. Unresolved.
- Attention-score vs. attention-free scoring. L2/K-norm (2406.11430, EMNLP 2024) and KeyDiff (2504.15364) score without materialising attention, which is the only practical option under fused attention kernels — but "The Pitfalls of KV Cache Compression" finds K-Norm among the methods that silently drop instructions. Cheapness and robustness are in tension here and the field has not settled it.
- Whether non-uniform budget allocation is real. PyramidKV (layer-wise), Ada-KV (2407.11550, head-wise) and LAVa (2509.09754) disagree on the allocation rule, and PyramidKV itself degenerates to SnapKV at aggressive ratios — several groups argue most of the reported gain comes from the observation window, not the allocation.
- Benchmark choice decides the winner. LongBench/NIAH rankings do not survive multi-turn cache reuse (SCBench, 2412.10319), instruction-following stress (2510.00231), or worst-case rather than mean aggregation (DefensiveKV, 2510.13334). Treat any single-benchmark ranking as an anecdote.

NAMED-BUT-NOT-INCLUDED (crowded out by the 9-slot cap, all worth a skim): ChunkKV (2502.00299) — chunk-granularity eviction, the closest analogue to cache-line rather than word granularity, plus cross-layer index reuse; FastKV (2502.01068) — token-selective propagation, decouples prefill compute reduction from KV budget; the L2-norm paper (2406.11430) — the origin of attention-free scoring; Ada-KV (2407.11550) — head-level budget allocation; SCBench (2412.10319) — the multi-turn/shared-context evaluation.

## The serving layer as a memory hierarchy

**must**

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180v1) (2023, `2309.06180v1`) — Read this for the founding OS analogy — block tables, internal vs external fragmentation of the KV cache, and copy-on-write sharing; every later serving paper assumes this vocabulary.
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079v4) (2024, `2407.00079v4`) — Read this for what the KV cache looks like once it becomes a first-class distributed store: DRAM/SSD tiering across the fleet, cache-aware scheduling, and prediction-based early rejection under overload — with production numbers, not a simulator.

**should**

- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104v2) (2023, `2312.07104v2`) — Read this for RadixAttention — prefix reuse as an LRU radix tree over the KV cache — and for why cache hit rate stops being an implementation detail and becomes a scheduling objective.
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670v3) (2024, `2401.09670v3`) — Read this for the goodput argument (TTFT and TPOT as separate SLOs) that justifies putting prefill and decode on different GPUs, and for the KV-transfer cost model that decides when the split actually pays.
- [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](https://arxiv.org/abs/2405.04437v3) (2024, `2405.04437v3`) — Read this for the counter-argument to paging: CUDA VMM APIs defragment physical memory while keeping the KV cache virtually contiguous, so attention kernels need no rewrite. The sharpest test of whether the OS-paging analogy is load-bearing or just familiar.
- [LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference](https://arxiv.org/abs/2510.09665v2) (2025, `2510.09665v2`) — Read this for the KV cache as a reusable layer *beneath* the engine — offload to CPU/disk/object store and cross-engine transfer for PD disaggregation — and for the engineering detail of what a tiering layer must actually implement.
- [TraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale](https://arxiv.org/abs/2512.18194v1) (2025, `2512.18194v1`) — Read this for CXL as the KV substrate: load/store access to a rack-wide prefix cache instead of RDMA copies, and — more instructive — the synchronization and consistency problems non-coherent shared memory forces you to solve.

**could**

- [Splitwise: Efficient generative LLM inference using phase splitting](https://arxiv.org/abs/2311.18677v2) (2023, `2311.18677v2`) — Read this for the hardware-heterogeneity angle DistServe leaves out: prefill is compute-bound and decode is bandwidth-bound, so they want different machines and different power envelopes — with per-phase provisioning numbers.

> **Recency and contested points.** Last ~6 months (Feb-Jul 2026) is dominated by two things, both new since the 2024 canon.

(1) A synthesis finally exists. "From Tensor Buffer to Distributed Memory Hierarchy" (arXiv 2607.02574, 30 Jun 2026) is the first survey to treat KV management explicitly as a distributed memory hierarchy rather than a bag of tricks. It classifies 30+ systems on four axes (locality, lifetime, ownership, substrate) into five archetypes — local-paged, disaggregated-pipeline, shared-store, memory-pool, hybrid-tier — and names seven measurement gaps: metadata-to-data ratio in shared stores, KV handoff granularity cost, prefix hit rate vs lookup overhead, cross-tier migration latency under eviction, ownership coordination overhead, multi-tenant isolation semantics, and durability contracts for persisted KV. For a lab whose stated weakness-of-the-literature is "reports that something helped without isolating which mechanism," that gap list is the most actionable page in the track.

(2) CXL-pooled KV went from one paper to a subfield, and it is entirely post-cutoff for most readers. TraCT (2512.18194, Dec 2025) is the anchor; since then: CXL-SpecKV (2512.11920), SAC: Disaggregated KV Cache System for Sparse Attention LLMs with CXL (2606.19746, Jun 2026), ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories (2606.12556, Jun 2026), Predictive Multi-Tier Memory Management for KV Cache (2604.26968, Apr 2026), and HyMCache (2607.18141, Jul 2026), which lays out an explicit G1-G4 hierarchy: HBM / system DRAM / local SSD / shared remote. I verified these ids exist as arXiv listings but did not read them closely enough to rank them, so they are cited here rather than placed in the list. Direct relevance to this lab's hardware: the Strix Halo unified-memory platform collapses the HBM/DRAM tier boundary these papers are built around, which is either a confound or an interesting natural experiment depending on how the arm is framed.

CONTESTED, three live disputes — do not let the reading list pick a side:

- Paging vs. contiguous virtual memory. vLLM's block table (2309.06180) buys defragmentation at the price of non-contiguous KV and custom attention kernels; vAttention (2405.04437, ASPLOS'25) argues CUDA VMM gets the same defragmentation with virtually contiguous KV and stock kernels. The 2026 survey still lists this as unresolved. The systems-engineering instinct ("of course you page") is exactly what is under dispute.

- Disaggregation vs. chunked prefill. DistServe/Splitwise split prefill and decode across machines; Sarathi-Serve (2403.02310, OSDI'24) argues stall-free chunked-prefill scheduling inside one pool captures most of the benefit without the KV-transfer cost. Both address the same phase asymmetry with different synchronization costs, and 2026 papers exist on both sides (e.g. 2508.01989 explicitly tries to unify them).

- Centralized vs. distributed KV ownership. Coordinator-based stores (Mooncake-style) simplify policy but bottleneck; peer protocols distribute load and add complexity. The survey flags ownership as the axis that most predicts a system's design, and as the least settled.

One structural note: the survey identifies an *empty* design point — per-session KV lifetime — despite chat/agent workloads obviously wanting session-scoped retention. That is a stated open hole in the taxonomy, not a paper recommendation.

## Constant-state memory — SSMs and linear attention

**must**

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752v2) (2023, `2312.00752v2`) — Read this for the core move that made constant-state models competitive: input-dependent (selective) gating, plus the hardware-aware scan that makes a sequential recurrence actually fast on a GPU — the systems half is as important as the math half.
- [Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927v1) (2023, `2312.04927v1`) — Read this for the diagnostic that tells you *where* constant state breaks — MQAR isolates multi-query associative recall and shows the failure is a state-capacity limit, not a training artifact. This is the benchmark your own eviction/compression work should be scored against.
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060v1) (2024, `2405.21060v1`) — Read this for the single mental model that unifies the whole track: attention and SSMs are two decompositions of the same structured matrix, so 'KV cache vs. fixed state' becomes one dial (state size) rather than two architectures.

**should**

- [Simple linear attention language models balance the recall-throughput tradeoff](https://arxiv.org/abs/2402.18668v2) (2024, `2402.18668v2`) — Read this for the explicit recall-vs-recurrent-state-size Pareto frontier: it treats state as a fixed memory budget you spend, and shows sliding-window + linear attention lets you slide along the curve. Closest thing in the literature to a capacity-planning argument.
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484v6) (2024, `2406.06484v6`) — Read this for the mechanism everything after 2024 builds on: the state as a linear associative memory that gets *corrected* (write-with-erase via the delta rule) instead of blindly accumulated, plus the Householder-product trick that makes it parallelizable over sequence length.
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464v3) (2024, `2412.06464v3`) — Read this for the erase/update decomposition — gating gives fast bulk forgetting, the delta rule gives targeted overwrite — and because it is the layer that actually shipped in production hybrids (Qwen3-Next, Olmo Hybrid), so it is the realistic baseline to ablate against.
- [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569v1) (2026, `2603.15569v1`) — Read this for the current SSM state of the art and its inference-first framing: complex-valued state updates for state tracking, MIMO for decode efficiency, and the headline capacity result — Mamba-2 perplexity at half the state size.

**could**

- [Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity](https://arxiv.org/abs/2607.07386v1) (2026, `2607.07386v1`) — Read this for the most systems-legible attack on the capacity wall: replace the dense outer-product state with sparse addressed reads/writes into a large explicit memory, and measure capacity vs. quality under an isoFLOP constraint. This is a memory hierarchy, spelled out.

> **Recency and contested points.** Substantial movement in the last ~6 months; a pre-2025 reading list for this track is now clearly incomplete.

New since ~Feb 2026 (all IDs seen on arxiv.org this session, but not all read in depth):
- Mamba-3 (2603.15569, Mar 2026, ICLR'26) — complex-valued state updates for state tracking + MIMO decode; claims Mamba-2 perplexity at half the state size. Included above.
- Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention (2605.22791, NVIDIA, May 2026) — splits Gated DeltaNet's single scalar beta into a channel-wise erase gate and a channel-wise write gate; reports the strongest aggregate result under matched param *and* matched state size vs. Mamba-2, GDN, KDA, and Mamba-3. If you ablate GDN, this is now the fairer baseline.
- Sparse Delta Memory (2607.07386, Meta FAIR, Jul 2026) — sparse addressing into a large explicit memory. Included above.
- A Hippocampus for Linear Attention / HOLA (2607.02303, Jul 2026) — semiparametric test-time memory: delta-rule compressive state *plus* a bounded exact KV cache, explicitly framed as complementary learning systems. This is the paper closest to a Mnemosyne-shaped contribution and is worth reading even though it is single-author and unreplicated.
- Kernelized Linear Attention: Breaking the Capacity Wall with Symmetric Cones (2607.17419, Jul 2026) — recasts associative recall as spherical packing and gives a Welch-bound interference floor for state capacity. Theory side of the MQAR story.
- Log-Linear Attention (2506.04761, ICLR 2026) — replaces the fixed-size state with a logarithmically growing hierarchy of states. Directly attacks this track's defining O(1) assumption; arguably the most important conceptual reframing of the last year and a near-miss for the list above.

CONTESTED — do not let anyone tell you this is settled. Two production-scale claims from well-resourced labs point in opposite directions:
- MiniMax abandoned hybrid linear attention for M2 (see the M2 technical report, 2605.26494, and MiniMax's own "Why Did M2 End Up as a Full Attention Model?" note). Their stated reasons: no efficient-attention variant reliably matched full attention on reasoning/coding/agentic tasks at scale; hybrids looked fine on standard benchmarks but showed clear deficits on multi-hop reasoning, and SWA variants degraded past 32K.
- Kimi Linear (2510.26692) claims the opposite under matched-scale pretraining — a 3:1 KDA/full-attention hybrid matching or beating full attention including in RL post-training.
Neither is a controlled academic ablation; both are shipping-product retrospectives with commercial incentives attached. Treat the disagreement as the open question of the track, not as evidence for either side. MiniMax's teased M3 reportedly pivots to sparse rather than linear attention, which is a third position again.

Requested-but-cut for space, still worth having in the bibliography: RWKV-7 "Goose" with Expressive Dynamic State Evolution (2503.14456 — the non-Mamba lineage, and the state-tracking-beyond-TC0 expressivity argument); xLSTM: Extended Long Short-Term Memory (2405.04517 — the sLSTM/mLSTM matrix-memory line from Hochreiter's group); MiniMax-01: Scaling Foundation Models with Lightning Attention (2501.08313 — the I/O-aware lightning-attention kernel and the 456B hybrid that MiniMax has since walked back).

## Hybrid architectures and ratio selection

**must**

- [A Systematic Analysis of Hybrid Linear Attention](https://arxiv.org/abs/2507.06457v2) (2025, `2507.06457v2`) — Read this for the actual ratio-selection evidence: 72 trained models sweeping six linear-attention variants against five hybridization ratios, landing on 3:1–6:1 linear-to-full and showing recall collapses as full-attention layers thin out.
- [Hybrid Architectures for Language Models: Systematic Analysis and Design Insights](https://arxiv.org/abs/2510.04800v3) (2025, `2510.04800v3`) — Read this for the two-axis map of the whole design space — inter-layer (sequential, Jamba-style) vs intra-layer (parallel, Hymba-style) fusion — scored on long context, scaling, and train/inference cost, with explicit design recipes.
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692v2) (2025, `2510.26692v2`) — Read this for the current reference point a hybrid must beat: 3:1 KDA-to-MLA interleave, 75% KV cache reduction and 6x decode throughput at 1M context, and the first fair-comparison claim of a hybrid outperforming full attention.

**should**

- [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887v2) (2024, `2403.19887v2`) — Read this for the KV-cache economics that justify hybrids in systems terms — 4GB attention cache at 256K context vs 32GB for Mixtral and 128GB for Llama-2-70B — plus the 1:7 attention-to-Mamba ratio that anchored everything after it.
- [Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling](https://arxiv.org/abs/2406.07522v3) (2024, `2406.07522v3`) — Read this for the cleanest statement of the division of labour your lab's SWA/global reference model relies on — recurrent state compresses history, sliding-window attention holds precise recent memory — at the simplest possible interleave.
- [Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models](https://arxiv.org/abs/2504.03624v4) (2025, `2504.03624v4`) — Read this for how a ratio survives contact with production scale: 8B and 56B/47B hybrids holding accuracy against Qwen-2.5 and Llama-3.1 at up to 3x faster inference, including pruning/distillation of a hybrid.
- [Rethinking the Role of Efficient Attention in Hybrid Architectures](https://arxiv.org/abs/2606.15378v1) (2026, `2606.15378v1`) — Read this for the counterintuitive result that reframes ratio picking: larger SWA windows delay retrieval-head formation ('Large-Window Laziness'), and the efficient-attention choice governs how fast long-context ability emerges rather than its ceiling.

**could**

- [Hymba: A Hybrid-head Architecture for Small Language Models](https://arxiv.org/abs/2411.13676v1) (2024, `2411.13676v1`) — Read this for the parallel (intra-layer) alternative to interleaving, where attention and SSM heads run side by side in one layer, stacked with cross-layer KV sharing and partial SWA for an 11.67x cache reduction.
- [Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing](https://arxiv.org/abs/2607.07953v1) (2026, `2607.07953v1`) — Read this for a unified-notation comparison of what actually goes in the linear half — DeltaNet, Gated DeltaNet, Kimi Delta Attention, Gated DeltaNet-2 — on expressivity, memory decay, and throughput, with hybrid vs pure runs at 350M/1.3B/3B.

> **Recency and contested points.** Two things changed in the last ~6 months and both matter for ratio selection.

1) 3:1 became the industry default, then immediately got contested. Kimi Linear (3:1 KDA:MLA) was followed by Qwen3-Next and then Qwen3.5-397B-A17B (Feb 2026), both roughly 3:1 Gated DeltaNet to full attention. But MiniMax M2.5 (Feb 2026) shipped as full attention with plain MHA, explicitly on reliability grounds. The convergence is real but not settled — do not treat 3:1 as a solved constant.

2) The field openly disagrees on whether the ratio sets a capability ceiling or just a training-efficiency knob. CONTESTED: "A Systematic Analysis of Hybrid Linear Attention" (2507.06457, revised June 2026) finds recall degrades sharply as full-attention density drops below 3:1, implying a real ceiling. "Rethinking the Role of Efficient Attention in Hybrid Architectures" (2606.15378, June 2026) argues the opposite reading — different hybrid configurations converge to comparable performance given enough training, so the efficient-attention design mainly controls how fast long-context capability emerges. Same year, same question, incompatible framings. Present both; the resolution likely depends on token budget, which is exactly the axis a small-scale ablation rig can attack.

Also new and worth knowing: a 2026 wave reframes hybrid design as post-hoc conversion/distillation from a pretrained transformer rather than a pretraining commitment — e.g. "Morphing into Hybrid Attention Models" (2606.30562, June 2026), which freezes weights and learns layerwise gates to pick which layers keep full attention under a fixed budget. If that line holds, ratio and layer-assignment search becomes cheap and moves after pretraining, which changes the cost structure of this entire question. Separately, "Olmo Hybrid: From Theory to Practice and Back" (2604.03444, Apr 2026, rev. Jun 2026) is the first fully-open 7B hybrid (Gated DeltaNet replacing sliding-window layers) and is the most reproducible baseline available, though it prescribes no ratio methodology.

One gap to flag: Zamba2 was requested but I could not establish a reliable arXiv id for it this session, so I excluded it rather than guess. Its shared-attention-block design is partly covered by the intra-layer analysis in 2510.04800. "Mechanistic Design and Scaling of Hybrid Architectures" (2403.17844, verified) is the historical origin of systematic hybrid topology search (500+ models, synthetic-task-driven) and is a reasonable tenth read if the reader wants the methodology lineage.

## Long-context behaviour and effective context

**must**

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v5) (2021, `2104.09864v5`) — Read this for the actual mechanics of RoPE — position injected as a rotation applied to queries and keys, giving relative-distance behavior with a built-in long-term decay. Load-bearing for Mnemosyne specifically: KV cache stores post-rotation keys, so any eviction, compaction, or re-packing scheme has to reason about whether it is silently changing a token's encoded position.
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172v3) (2023, `2307.03172v3`) — Read this for the U-shaped position bias result — accuracy is highest when the needed token sits at the start or end and degrades sharply in the middle. For a memory subsystem this is the direct justification for position-aware eviction: KV entries are not equally reachable, so uniform-value cache policies are already wrong before you write any code.
- [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654v3) (2024, `2404.06654v3`) — Read this for the measurement methodology that separates advertised from effective context: 13 synthetic tasks across retrieval, multi-hop tracing, aggregation, and QA, showing models with near-perfect needle-in-a-haystack scores collapsing well below their claimed length. This is the harness design you copy when your own long-context claims need to survive scrutiny.

**should**

- [The Impact of Positional Encoding on Length Generalization in Transformers](https://arxiv.org/abs/2305.19466v2) (2023, `2305.19466v2`) — Read this for the NoPE result and the controlled head-to-head of APE, T5-relative, ALiBi, RoPE, and no positional encoding on length generalization. The useful takeaway is negative: RoPE and ALiBi are not chosen because they extrapolate well, so do not assume your positional scheme is the thing buying you long context.
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071v3) (2023, `2309.00071v3`) — Read this for the frequency-band view of RoPE extension — why naive linear interpolation crushes high-frequency dimensions and how NTK-aware/wavelength-dependent scaling plus attention temperature fixes it at 10x fewer tokens. Also the cleanest single place to learn Position Interpolation (2306.15595) and NTK-aware scaling, since it explains and benchmarks both.
- [Positional Biases Shift as Inputs Approach Context Window Limits](https://arxiv.org/abs/2508.07479v1) (2025, `2508.07479v1`) — Read this for the correction to the naive lost-in-the-middle story: the U-shape only holds up to roughly 50% context occupancy, after which primacy decays, recency persists, and the bias becomes distance-based. Directly changes eviction-policy design, because the position prior you exploit depends on how full the window is.
- [LongBench Pro: A More Realistic and Comprehensive Bilingual Long-Context Evaluation Benchmark](https://arxiv.org/abs/2601.02872v1) (2026, `2601.02872v1`) — Read this for the current-scale confirmation that the RULER gap never closed: 46 models, 8k-256k, naturally occurring rather than synthetic tasks, and effective context still falls short of claimed with language-specific gaps. Cite this rather than RULER when you need a 2026-era number for the effective-vs-advertised claim.

**could**

- [How to Train Long-Context Language Models (Effectively)](https://arxiv.org/abs/2410.02660v4) (2024, `2410.02660v4`) — Read this for the actual long-context extension training recipe with ablations — data mix (code repos and books, but mixed with short high-quality data), training beyond your evaluation length, and short-instruction SFT sufficing. Also read it for the evaluation stance: they explicitly reject perplexity and bare NIAH as progress signals, which is the standard your own ablations should meet.
- [ATLAS: All-round Testing of Long-context Abilities across Scales](https://arxiv.org/abs/2605.28079v1) (2026, `2605.28079v1`) — Read this for the two failure modes at frontier scale out to 1M tokens: performance collapses as length grows, and strong retrieval does not transfer to downstream use. The rank-instability finding (7 models shifting 2+ ranks, gaps up to 12 positions between the 8K-128K and 8K-1M regimes) is the concrete argument against reporting a single headline long-context score.

> **Recency and contested points.** Substantial movement in this track since late 2025 — a list stopping at RULER (2024) would be materially out of date.

WHAT IS NEW (last ~6 months, all checked this session):
- LongBench Pro (arXiv 2601.02872, Jan 2026) and ATLAS (arXiv 2605.28079, May 2026) are the current effective-vs-advertised references, and both are included above. ATLAS matters most because it pushes to 1M and reports rank instability between length regimes; LongBench Pro matters because it uses naturally occurring rather than synthetic tasks across 46 models. RULER is now best cited as the methodology that started this, not as a current measurement.
- "A Structural Theory of Position Bias in Transformers" (arXiv 2602.16837, Feb 2026, rev. May 2026) derives U-shaped influence profiles from causal masking plus residual connections via residual-aware cumulative attention rollout. Did not make the nine, but it is the best current mechanistic account of why lost-in-the-middle exists, and worth reading directly after 2307.03172 if this lab wants an attribution story rather than an outcome story.
- "Jet-Long: Efficient Long-Context Extension with Dynamic Bifocal RoPE" (arXiv 2607.07740, July 2026) is the freshest RoPE-extension work — tuning-free, length-adaptive rescaling paired with a RoPE-faithful local window. Too new to anchor a curriculum on, but it is where YaRN's line of work has gone.
- Chroma Research's "Context Rot: How Increasing Input Tokens Impacts LLM Performance" (Hong, Troynikov, Huber, July 2025) is widely cited in this track but is an industry tech report, not an arXiv paper — I found no arXiv id for it and deliberately did not invent one. Its counterintuitive claim, that coherent well-structured input degrades attention more than shuffled input, is worth knowing even though the venue is weaker.

WHAT IS CONTESTED (presented as contested, not resolved):
1. Whether position bias is architectural or trainable. The structural-theory line (2602.16837) and commentary around context rot argue it falls out of causal masking, residual connections, and RoPE's long-term decay — i.e. not fixable by more long-context training. The calibration line ("Found in the Middle", arXiv 2406.16008) reports recovering up to 15 points by correcting attention bias at inference, which implies it is at least substantially correctable. Do not let a curriculum section assert either.
2. Long context vs. RAG. arXiv 2501.01880 finds long context generally beats RAG on QA benchmarks while RAG wins on dialogue and general queries; other 2025-2026 work reports the reverse ordering as corpus size grows. The honest summary is that the ordering is task- and scale-dependent and the cost asymmetry is large, not that one has won.
3. Whether the lost-in-the-middle U-shape is even the right model. 2508.07479 shows it holds only below ~50% context occupancy and becomes distance-based above that, so papers reporting a clean U-shape and papers reporting recency-dominance may simply be probing different occupancy regimes rather than disagreeing.

CAVEAT ON DATES: several 2026 arXiv ids above (26xx.xxxxx) were confirmed by fetching the arXiv abstract pages this session, but they are recent enough that version numbers and venue status may still move.

## Agent memory and its failure modes

**must**

- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560v2) (2023, `2310.08560v2`) — Read this for the original virtual-memory/paging analogy applied to context windows — main context vs. external context, self-editing memory via tool calls, and page-fault-style interrupts. This is the vocabulary every later agent-memory paper assumes, and it maps 1:1 onto storage-hierarchy intuition you already have.
- [A Survey on Long-Term Memory Security in LLM Agents: Attacks, Defenses, and Governance Across the Memory Lifecycle](https://arxiv.org/abs/2604.16548v2) (2026, `2604.16548v2`) — Read this for the six-phase memory lifecycle threat model — it shows poisoning as a cross-phase chain (write -> persist -> propagate -> resist-cleanup) rather than a single injection event. Closest thing to a threat-modeling doc for a memory subsystem, and the right frame if Mnemosyne ever accepts untrusted writes.

**should**

- [AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases](https://arxiv.org/abs/2407.12784v1) (2024, `2407.12784v1`) — Read this for the concrete mechanism: triggers optimized so poisoned entries occupy a distinct embedding region, giving >80% attack success at <0.1% poison rate with <1% benign degradation. Grounds the surveys in an actual retrieval-level exploit.
- [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110v11) (2025, `2502.12110v11`) — Read this for the alternative to MemGPT's fixed tiers: Zettelkasten-style notes with generated keywords/tags, dynamic linking, and retroactive refinement of older memories on write. The contrast with MemGPT is the core design axis — fixed hierarchy vs. self-organizing index.
- [From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents](https://arxiv.org/abs/2606.04329v2) (2026, `2606.04329v2`) — Read this for the taxonomy of memory *write channels* — including compaction-driven writes, which means your summarizer is an attack surface. It is the paper that tells you which code path to instrument, not just that poisoning exists.

**could**

- [Remembering More, Risking More: Longitudinal Safety Risks in Memory-Equipped LLM Agents](https://arxiv.org/abs/2605.17830v1) (2026, `2605.17830v1`) — Read this for the measurement that memory-induced violation rates rise monotonically with exposure length — cross-session contamination as a slow accumulation defect, not a discrete attack. Argues that any memory eval run over a short horizon will systematically under-report harm.
- [Parallel Context Compaction for Long-Horizon LLM Agent Serving](https://arxiv.org/abs/2605.23296v1) (2026, `2605.23296v1`) — Read this for compaction treated as a serving problem — sequential summarization blocks inference, so overlap it and control how much you compact. This is the agent-memory paper written in your native language of latency, blocking, and throughput.

> **Recency and contested points.** Substantial movement in the last ~6 months (Feb-Jul 2026), and this track is now unrecognizable from its 2023-24 state. (1) SECURITY WENT MAINSTREAM. The dedicated security survey 2604.16548 (Apr 2026, revised Jun 2026) did not exist before this window; its v1 carried a different title ("A Survey on the Security of Long-Term Memory in LLM Agents: Toward Mnemonic Sovereignty"), so cite the v2 title. Alongside it: the write-channel study 2606.04329 (Jun 2026), plus a cluster of sleeper/delayed-trigger work — "Hidden in Memory: Sleeper Memory Poisoning in LLM Agents" and "Plant, Persist, Trigger: Sleeper Attack on Large Language Model Agents" — where a benign-looking write lies dormant across sessions before activating. Secondary sources report OWASP added "Memory and Context Poisoning" as ASI06 in the 2026 Agentic AI Top 10; I did not verify that against OWASP directly. (2) FRESH SURVEYS SUPERSEDE THE OLD ONES. Besides 2605.06716, there is "Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers" (2603.07670, Mar 2026). Anything citing only the 2024 agent-memory surveys is stale. (3) CONTESTED — is agentic memory actually memory? "Contextual Agentic Memory is a Memo, Not True Memory" (2604.27707, seen in search results, id NOT fetch-verified) argues current systems are externalized note-taking, not memory in any consolidative sense; the MemOS/MemCube line argues the opposite, that memory becomes real once it is a scheduled resource spanning plaintext, activation, and parameter tiers. Both positions are live; do not present either as settled. (4) CONTESTED — where the control plane belongs. "Control-Plane Placement Shapes Forgetting: An Architectural Study of Agent Memory Across Thirteen System Configurations" (2606.15903, Jun 2026, verified) finds that *where* the LLM sits in the memory pipeline determines which failure modes are even addressable, with mutation-time placement winning. This directly contradicts the common assumption that retrieval-time reranking is where the leverage is — relevant if Mnemosyne has to choose a plug point. (5) CONTESTED — do defenses work? Secondary summaries claim five of six defense classes fail against delayed-trigger attacks and only tool-layer memory restriction holds structurally. I could not confirm that specific claim in a primary source; treat it as a hypothesis worth testing, not a finding. (6) COVERAGE GAP I DELIBERATELY LEFT OUT: "A Survey of Context Engineering for Large Language Models" (2507.13334, Jul 2025, verified) is the canonical context-engineering survey but predates this window and treats memory as one subsection; I chose the compaction-as-serving paper instead because it is both newer and closer to this lab's bottleneck. Also omitted for space: Mem0 (2504.19413) for the production-deployment angle, and the benchmark line (MemoryAgentBench 2507.05257, plus 2026 benchmarks adding forgetting-aware metrics) — worth a follow-up pass if evaluation methodology becomes the question.

## MoE routing, scaling laws, hyperparameter transfer

**must**

- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556v1) (2022, `2203.15556v1`) — Read this for the matched-budget discipline your ablation rig depends on: the three IsoFLOP estimation approaches, and why a params-vs-tokens allocation must be stated before any arm is run rather than defended afterward.
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664v1) (2024, `2408.15664v1`) — Read this for the mechanism isolated from a frontier-model launch: the bias-update rule, its hyperparameter, and controlled 1B/3B ablations — plus the batch-level vs global-level load-balance distinction that explains why aux-loss numbers look better than they are.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437v2) (2024, `2412.19437v2`) — Read this for the complete modern MoE recipe in one document — sigmoid gating, per-expert bias for aux-loss-free balancing, shared + fine-grained experts, node-limited routing, and the sequence-wise balance loss they kept anyway; it is the closest published analogue to the Laguna-class reference model.

**should**

- [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906v2) (2022, `2202.08906v2`) — Read this for the failure-mode catalog written like a systems postmortem — router z-loss, bf16/precision-driven divergence, capacity-factor behavior, expert-count sweeps, and the train/fine-tune gap; it names most of the ways an MoE run silently goes wrong.
- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466v2) (2022, `2203.03466v2`) — Read this for the reason your 20M-300M ablations can say anything about larger models at all — muP makes the optimal LR stable across width, so an arm comparison is not just a comparison of how well each arm happened to be tuned.
- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066v1) (2024, `2401.06066v1`) — Read this for why the expert topology looks the way it does: fine-grained expert segmentation plus shared-expert isolation as the two levers against knowledge redundancy, and for their expert-specialization measurements — the closest thing to an operational definition of expert collapse.
- [Problems with Chinchilla Approach 2: Systematic Biases in IsoFLOP Parabola Fits](https://arxiv.org/abs/2603.22339v3) (2026, `2603.22339v3`) — Read this before you fit a single IsoFLOP curve: the standard parabola fit is biased even on noise-free synthetic data, the bias comes from grid width, off-center sampling, and loss-surface asymmetry, and Approach 3 with variable projection is the recommended replacement.

**could**

- [Sigmoid Gating is More Sample Efficient than Softmax Gating in Mixture of Experts](https://arxiv.org/abs/2405.13997v3) (2024, `2405.13997v3`) — Read this for the actual argument behind a proteus-moe-sigmoid arm — sigmoid removes the inter-expert coupling softmax imposes and gives faster expert-estimation rates; note the analysis is small-scale regression theory, so it motivates the ablation rather than settling it.
- [When Are Experts Misrouted? Counterfactual Routing Analysis in Mixture-of-Experts Language Models](https://arxiv.org/abs/2605.07260v1) (2026, `2605.07260v1`) — Read this for evidence that a balanced router is not a good router: holding the model frozen and sampling equal-compute alternative routes shows the trained top-k choice is near-optimal on confident tokens and close to uninformative on the hard ones.

> **Recency and contested points.** Last ~6 months (Feb-Jul 2026), three things changed and a 2026 reader should not miss them.

(1) Scaling laws moved from "find a new law" to "the fitting methodology is wrong." Czech et al., "Problems with Chinchilla Approach 2" (2603.22339, Mar 2026, verified) shows the parabolic IsoFLOP fit is systematically biased even on noise-free data, quantifies ~6.5% budget misallocation on published Llama-3 IsoFLOP data, and rehabilitates Approach 3 via variable projection. This lands directly on a small-scale lab: the bias sources it names (narrow grid, uncentered sampling, loss-surface asymmetry) are exactly the conditions of a 20M-300M sweep. I found no single accepted "Chinchilla successor" law — the field has fragmented into conditional laws (data quality, optimizer choice, MoE sparsity, expert-vs-attention FLOP split) rather than one replacement, so treat any claim of a successor as contested.

(2) muP now has a practical superset. "Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration" (arXiv 2512.22382, Apple, ICLR 2026 — verified) extends transfer beyond width to depth, batch size, and training duration, and adds per-module HPs, reporting direct transfer to a ~14000x larger FLOP budget. MoE-specific: "$\mu$-Parametrization for Mixture of Experts" (arXiv 2508.09752 — verified) exists because routing and sparsity sit outside classic muP theory; if you are muP-ing a Proteus MoE arm, plain Tensor Programs V is not sufficient on its own.

(3) Aux-loss-free balancing acquired theory and a live dispute. "A Theoretical Framework for Auxiliary-Loss-Free Load Balancing of Sparse Mixture-of-Experts in Large-Scale AI Models" (arXiv 2512.03915, Dec 2025 — verified) recasts the bias-update rule as a primal-dual method with monotonic-improvement and logarithmic-regret results, validated on 1B DeepSeekMoE.

CONTESTED, do not pick a side: whether the bias trick is sufficient alone. DeepSeek's own result claims lower perplexity and 10-20x better global load balance than auxiliary loss. Multiple 2025-26 reports counter that under very high sparsity it leaves lower layers imbalanced, and that a small-weight auxiliary loss retained alongside the bias outperforms either alone. Treat the auxiliary-loss weight as an ablation axis, not a settled default. Similarly contested: whether sigmoid routing wins generally or only at high expert granularity.

Methodology caveat: web-search snippets surfaced a number of 2026 arXiv identifiers I could not confirm against arxiv.org within this session (several MoE-routing and MoE-scaling-law entries). Every id in the papers list above was confirmed by fetching its arxiv.org abstract page today; unconfirmed candidates were excluded rather than guessed. Two verified-but-omitted items worth queueing: "Scaling Laws for Fine-Grained Mixture of Experts" (2402.07871) for the granularity axis, and "Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models" (2501.12370) for choosing sparsity under a fixed budget.

## Rejected candidates

Recorded rather than deleted: knowing a citation failed verification is worth more later than a tidy list.

| Claimed title | Claimed id | Outcome |
|---|---|---|
| Rethinking Memory in AI: Taxonomy, Operations, Topics, and Future Directions | `2505.00675` | **Title drift, not a bad citation.** The id is correct; the paper was retitled to *Rethinking Memory in LLM based Agents: Representations, Operations, and Emerging Topics* in a later version, and is included above under that title. Cite by id — titles move. |
