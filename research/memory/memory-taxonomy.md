---
title: Five things called memory — fixing the vocabulary
version: 1.0.0
track: research/memory (note 1 of 10)
---

# Five things called memory

This note settles the vocabulary problem that blocks every other note in the track: the
word "memory" is used for five mechanically unrelated things, and two large literatures
apply *incompatible* names to *the same bytes*. It gives each of the five a definition
grounded in what it holds, what a read and a write physically do, what it costs, and
what it structurally cannot do — then names the single property that actually partitions
them, which is not capacity or speed but **whether the information exists anywhere
else**. It closes with the house vocabulary Chiron will use, so that later notes and
experiment arms can be read without guessing which camp a term came from.

---

## 1. Why the vocabulary is broken

Two research communities converged on the same word from opposite directions and never
reconciled.

The **serving community** arrived from systems. Its objects are tensors with a lifetime
and an owner; its vocabulary is blocks, pages, prefixes, eviction, tiering, hit rate.
The 2026 synthesis survey classifies thirty-plus KV systems on four axes — locality,
lifetime, ownership, substrate — and calls the thing a *tensor buffer* in its own title
`[C]` (arXiv 2607.02574, Jun 2026).

The **agent-memory community** arrived from cognitive science. Its objects are episodic,
semantic, procedural, working and long-term memories; its vocabulary is consolidation,
forgetting, reflection, retrieval. MemOS partitions memory into *parametric /
activation / plaintext* and explicitly places the KV cache in the "activation" tier as a
first-class scheduled resource `[C]` (arXiv 2507.03724, 2025).

Those are the same bytes. One camp calls them a buffer with an owner; the other calls
them a memory tier with a scheduler. They cross-cite rarely. A July 2026 paper is the
first serious attempt to put all of it under one frame — rate-distortion: every
compaction method, from KV eviction to prompt pruning to bounded recurrent state to
agent-memory consolidation, is one decision about what context-derived information to
keep at what fidelity under a budget `[C]` (arXiv 2607.08032, Jul 2026). It is three
weeks old and has settled nothing yet. Treat the unification as an open proposal, not
as the field's position.

Worse than two names for one thing: **one name for two different things.** "Working
memory" in the agent literature normally means text currently in the prompt window; in
the serving literature the closest analogue of a working set is the resident KV cache —
which is derived from that text but is a different object with a different size, a
different lifetime and a different failure mode. "Long-term memory" means model weights
to the cognitive camp and a vector store to the systems camp. A 2026 paper titles a
persistent quantized KV cache "Agent Memory Below the Prompt" `[C]` (arXiv 2603.04428,
Feb 2026) — the collision is not hypothetical, it is in titles.

---

## 2. The five, side by side

| | **Weights** (parametric) | **Recurrent state** (SSM/linear-attn) | **KV cache** (attention working set) | **Retrieval index** (RAG) | **Session store** (agent memory) |
|---|---|---|---|---|---|
| Holds | statistical regularities + facts absorbed from training data | a lossy running summary of the whole prefix | one K and one V vector per token per layer, exactly as computed | documents + an embedding index over them | facts, preferences, trajectories, distilled experience |
| Written by | gradient descent over the whole model | the forward pass, destructively, in place | the forward pass, append-only | an offline/async ingestion job | the agent, at runtime, from untrusted input |
| Read by | every forward pass, implicitly | one matrix read per token | softmax-weighted sum over *all* resident entries | ANN top-k, then text injected into the prompt | query → retrieve → inject, same as RAG |
| Size | fixed at O(params) | fixed at O(state), independent of tokens | O(layers x kv-heads x head-dim x tokens) | O(corpus) | O(interaction history), unbounded |
| Cost driver | training FLOPs; then free at inference | bandwidth of one small state read | **memory bandwidth at decode** | index + embedding + an extra prompt | storage + retrieval + prompt tokens |
| Lifetime | the checkpoint | one sequence | one request (or a shared prefix) | forever, external | across sessions, forever |
| Recoverable if deleted? | only by retraining | by recomputing from tokens | **exactly, by recomputing from tokens** | by re-indexing the corpus | **no — this is the only authoritative tier** |

That last row is the load-bearing one. Return to it in §8.

---

## 3. Weights — parametric memory

**What it holds.** Everything the model learned during training, in superposition, with
no index and no schema. There is no row for "Paris is the capital of France"; there is a
distributed pattern that makes that continuation likely.

**Capacity.** Two independent measurements, and they measure different quantities — do
not average them. Allen-Zhu and Li fit synthetic (name, attribute, value) tuples and
report a consistent **~2 bits of knowledge per parameter** across architectures `[C]`
(arXiv 2404.05405, 2024). Morris et al. train hundreds of models from 500K to 1.5B
params and put the capacity for *unintended memorization* of random data at **~3.6 bits
per parameter**, with a sigmoidal law and a grokking transition once capacity fills
`[C]` (arXiv 2505.24832, 2025). The first is useful-knowledge density; the second is a
storage upper bound. For the models this lab trains — 20M to 300M params — that is a
total parametric store of roughly **5–135 MB**, which is a startling number to hold next
to a KV cache measured in gigabytes.

**Read/write semantics.** Reads are implicit, associative and non-enumerable — you
cannot list what a model knows. Writes are gradient steps: **non-local** (a single fact
changes weights everywhere), **non-atomic**, and with no transaction, no rollback and no
read-your-writes guarantee.

**What it fundamentally cannot do.**
- *No addressable write at inference time.* Targeted editing exists — ROME localizes a
  fact to an MLP and rewrites it `[C]` (arXiv 2202.05262, 2022); MEMIT batches thousands
  of such edits `[C]` (arXiv 2210.07229, 2022) — but it does not hold up. Edits fail to
  propagate to logical consequences of the edited fact `[C]` (arXiv 2307.12976, 2023),
  and applying edits sequentially at scale produces gradual then catastrophic forgetting
  `[C]` (arXiv 2401.07453, 2024).
- *No delete.* There is no operation that removes a fact and leaves the rest intact.
- *No provenance.* You cannot ask which training document a belief came from.

**Systems bridge, and where it breaks.** It behaves like a lossily-compressed read-only
image — a compiled binary, not a database. The analogy breaks because a binary has
addresses: you can patch one byte and know exactly what changed. Here the "patch" is a
gradient step whose blast radius is the entire parameter vector, and the ripple-effect
and forgetting results above are the empirical measurement of that blast radius.

---

## 4. Recurrent state — SSM and linear-attention memory

**What it holds.** One fixed-size tensor per layer per sequence, carrying a lossy summary
of everything seen so far. Mamba-2's `ssm_state` is allocated with shape
`(batch, nheads, headdim, d_state)` — **no sequence-length term at all**, verified in the
source at `architecture/mamba/mamba_ssm/modules/mamba2.py:352` (`CODE_MAP.md`). Decode
footprint at 1K tokens and 1M tokens is byte-identical.

**Cost model, written out.** Per layer, per sequence:

```
state_bytes = n_heads x d_head x d_state x p
```

`n_heads` = heads in that layer; `d_head` = channels per head; `d_state` = the
state-space dimension (Mamba-2 defaults `d_state=128`, `headdim=64`); `p` = bytes per
element. Nothing on the right-hand side is a function of token count. That is the whole
selling point, and it is also the whole limitation.

For a Gated DeltaNet head the state is one `d_k x d_v` matrix, and each token performs
the same three-step read-modify-write against it: decay the entire matrix by a learned
scalar `exp(g) in (0,1]`, read back what it currently returns for this token's key
(`h^T k`), then add the rank-1 outer product of `k` with `beta * (v - h^T k)`. Verified
line by line at `architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:54,56,58`
(`CODE_MAP.md`). `beta = 1` is an exact overwrite of that key's direction, `beta = 0` a
no-op.

**Read/write semantics.** Destructive read-modify-write, in place. There is no slot per
token, so no entry can be individually addressed, evicted, or refreshed.

**What it fundamentally cannot do.**
- *Cannot recall a specific past token exactly.* The carry is overwritten by a decay
  multiply every step (`states = scale * states + new_states`); token 5's contribution is
  not recoverable at token 5000 because it was destroyed, not relocated. There is no
  lower tier to page it back from.
- *Cannot forget selectively, in the shipped configuration.* Gated DeltaNet's decay is a
  single scalar applied to the whole K x V matrix — every stored association is
  attenuated by the same factor each step. The fused kernel implements per-key and
  per-channel gates (`USE_GK`/`USE_GV`) and the layer simply does not pass them
  (`CODE_MAP.md`). "Gating = selective forgetting" is backwards: the gate is
  indiscriminate decay; the *delta* term is the targeted erase.
- *Cannot be grown at inference time.* Capacity is a training-time architectural choice.
- *Its failure mode is interference, not a miss.* Keys are L2-normalized continuous
  vectors, so a *similar* key partially clobbers a neighbour. The model cannot detect
  that this happened. MQAR (multi-query associative recall) is the diagnostic that
  isolates this as a state-capacity limit rather than a training artifact `[C]`
  (arXiv 2312.04927, 2023), and the recall-vs-state-size Pareto frontier is the
  capacity-planning statement of the same fact `[C]` (arXiv 2402.18668, 2024).

**One more caveat that costs people money:** "constant state" is a **decode-time**
property only. During training and prefill the chunked scan materializes one fp32 state
per chunk before the boundary pass runs, so activation memory there is `O(L / chunk_size)`,
not `O(1)` (`CODE_MAP.md`, `ssd_combined.py:375`).

**Systems bridge, and where it breaks.** It looks like a fully-associative cache with a
global TTL tick and `beta` as write strength. It is emphatically **not a write-ahead
log**: a KV cache is an append-only exact record you can re-scan; this is a destructive
rolling aggregate with no replay.

---

## 5. KV cache — the attention working set

**What it holds.** For every token, in every layer that keeps a cache, one key vector and
one value vector, stored exactly as computed. Nothing is summarized. Nothing is merged.

**Capacity arithmetic, every symbol translated.**

```
kv_bytes = 2 x L x n_kv x d_head x p x T
```

`2` because each token contributes both a K and a V vector. `L` = number of layers
holding a cache. `n_kv` = key/value heads per layer (this is the GQA knob: multi-head
attention sets `n_kv` = number of query heads, MQA sets it to 1, GQA to something in
between) `[C]` (arXiv 2305.13245, 2023). `d_head` = channels per head. `p` = bytes per
element (2 for bf16, 1 for FP8). `T` = tokens resident.

For the lab's reference model, from the config read off the artifact rather than quoted
`[M]` (2026-07-26, `ASSUMPTIONS.md: kv-per-token-laguna`): `L=48`, `n_kv=8`,
`d_head=128`, `p=2` gives `2 x 48 x 8 x 128 x 2 = 196,608 bytes = 192 KiB per token`, so
**24 GiB at 128k context** — if every layer were global. It isn't: 36 of 48 layers are
sliding-window at 512 tokens, set by one line, `config.layer_types[layer_idx]`
(`CODE_MAP.md`, `modeling_laguna.py:365`). Redoing the arithmetic with 12 global layers
at `T` and 36 windowed layers capped at 512:

```
kv_bytes = 2 x n_kv x d_head x p x (12*T + 36*512)
         = 4096 x (12*131072 + 18432)  bytes  at T = 128k
         = 6.07 GiB
```

A 4x reduction from one config list. One clarification, because the register briefly had
this wrong: `laguna-heads-uniform` is refuted for **query** heads only — they are read
per layer from `config.num_attention_heads_per_layer` (48 full / 72 sliding) — while
`num_key_value_heads` is uniform at 8 with no per-layer override `[M]` (verified against
`config.json` at `b0a9fd7c850e`, 2026-07-26). Query heads do not appear in the KV
product, so **192 KiB/token is exact.** What varies per layer is the GQA group size
`G = n_q/n_kv` (6 full, 9 sliding), which moves decode arithmetic intensity, not bytes.

**Why decode is bandwidth-bound, with our own numbers.** Consider one decoded token at
batch size 1 with multi-head attention. Per layer you load `T x n_kv x d_head` K elements
and the same number of V elements, and you perform 2 FLOP per K element (the query dot
product) and 2 FLOP per V element (the weighted sum). In bf16 that is 4 FLOP per 4 bytes:
**1 FLOP per byte**. With GQA at group size `g = n_q / n_kv`, FLOPs scale with query
heads while bytes scale with KV heads, so intensity becomes **`g` FLOP/byte** — Laguna's
`g = 6`.

Our machine balance, both numbers measured on this box: `20.9 TFLOP/s` bf16 GEMM at
8192³ and `~200 GB/s` sustained inside the fast tier `[M]` (2026-07-26,
`scripts/benchmark_gemm.py`, `notebook/uma-carveout-controls-fast-tier.md`). Balance
point = `20.9e12 / 200e9 ≈ 105 FLOP/byte`. Attention decode delivers 1 (MHA) to 6 (GQA
at g=6). **Attention decode on this machine runs at roughly 1–6% of peak arithmetic**;
it is a pure bandwidth problem, which is the single mental model everything else in this
track depends on `[C]` (arXiv 1911.02150, 2019). Note the corollary that is usually
missed: GQA does not only shrink bytes, it *raises arithmetic intensity by the same
factor* — two wins from one knob.

**Read/write semantics.** Writes are append-only, one entry per token per layer, by the
forward pass. Reads are **not lookups**: there is no address and no key equality test.
Every decode step performs a softmax-weighted sum over the entire resident set, per head,
per layer. You cannot read one entry; you read all of them, weighted.

**What it fundamentally cannot do.**
- *Cannot hold anything not in the token sequence.* It is a function of the tokens and
  the weights. It adds no information.
- *A hit removes compute, not I/O, and never supplies information.* Evicting a KV block
  is never data loss, only a recompute — which is why Mooncake's `offload_force_evict`
  can throw bytes away rather than block on writeback, "a tradeoff no real storage tier
  is allowed to make" (`CODE_MAP.md`).
- *Cannot fault.* There is no miss path. In vLLM, `allocate_slots` returning `None` is not
  a fault to be serviced but an admission rejection that preempts the whole request —
  **eviction granularity is the sequence, not the page** (`CODE_MAP.md`,
  `block_pool.py`, `single_type_kv_cache_manager.py:97`).
- *Cannot be matched associatively.* Prefix reuse keys a block on
  `hash(parent_hash, tokens)` — a chain — so it is strictly prefix-ordered: the same 16
  tokens at a different offset are a different key, one changed token at position 0
  invalidates every downstream hash, and the match loop breaks at the first miss because
  a later hit is impossible by construction (`CODE_MAP.md`, `kv_cache_utils.py:596`).
- *Grows linearly and without bound.* This is the whole reason the eviction, compression,
  and tiering literatures exist `[C]` (arXiv 2412.19442, 2024; arXiv 2607.02574, 2026).

**Systems bridge, and where it breaks.** It is a buffer cache — with no backing store.
Every reflex you own about caches assumes a slower authoritative tier behind them;
here there is none, and "reload" means re-running prefill.

---

## 6. Retrieval index — RAG

**What it holds.** A document corpus plus a derived index (usually dense embeddings, often
hybridized with BM25). The original formulation couples a parametric seq2seq model to a
non-parametric dense index and retrieves passages to condition generation `[C]`
(arXiv 2005.11401, 2020).

**Cost model.** Ingestion is `O(corpus)` embedding FLOPs once, plus index build. Query
is one embedding forward pass plus an ANN search, then — the part that dominates in
practice — the retrieved passages are **prepended as tokens**, so every retrieval turns
into prefill compute and KV-cache bytes at the rate derived in §5. RAG is not free
memory; it converts storage into context-window occupancy.

**Read/write semantics.** Writes are an external, offline, transactional operation on a
normal database. Reads are top-k by vector similarity. The critical structural fact: **the
model never touches the store.** Retrieval is a prompt-construction step that happens
strictly before the forward pass.

**What it fundamentally cannot do.**
- *Cannot iterate on a query the model has not yet formulated.* In a single-shot pipeline
  retrieval precedes reasoning. Agentic/multi-hop loops mitigate this by paying more
  round trips; they do not remove the ordering constraint.
- *Cannot guarantee the retrieved text is used*, or that it wins against a conflicting
  parametric belief. Which source wins is an open attribution question, not a
  configuration setting.
- *Cannot represent what was never written down.*
- *Does not change the model's priors.* It changes the prompt.

**CONTESTED — RAG vs long context.** One evaluation finds long context generally beating
RAG on QA benchmarks while RAG wins on dialogue and general queries `[C]`
(arXiv 2501.01880, 2025); other 2025–2026 work reports the ordering reversing as corpus
size grows, and the cost asymmetry is large in RAG's favour. The honest 2026 summary is
that the ordering is **task- and scale-dependent**, and that hybrid pipelines are the
practical default. Do not let a curriculum section assert a winner. `[A]` low-to-medium
confidence that the ordering is stable within any single task family; the cheapest test
that would move it is a fixed-corpus sweep holding the generator constant while varying
corpus size, which is affordable at our scale.

---

## 7. Session store — agent memory

**What it holds.** Whatever the agent decided to write down: user preferences, extracted
facts, task trajectories, distilled procedures. MemGPT established the framing everything
else inherits — main context vs external context, self-editing memory via tool calls,
page-fault-style interrupts `[C]` (arXiv 2310.08560, 2023). A-MEM offers the contrasting
design: self-organizing Zettelkasten notes with generated links and retroactive refinement
instead of fixed tiers `[C]` (arXiv 2502.12110, 2025). The current surveys are
arXiv 2603.07670 (Mar 2026) and arXiv 2602.06052 (Feb 2026), which organizes the space by
substrate x cognitive mechanism x subject `[C]`.

**Cost model.** Storage is cheap; the real cost is the same as RAG's — every recalled item
becomes prompt tokens and therefore prefill and KV bytes. Plus a write-path cost that
RAG does not have: summarization/consolidation is itself an LLM call, and doing it
sequentially blocks inference `[C]` (arXiv 2605.23296, 2026).

**Read/write semantics.** Read-write, durable, cross-session, and — uniquely — **the
writes originate from model output and from untrusted input.**

**What it fundamentally cannot do.**
- *Cannot be recomputed.* Delete it and the information is gone. It is the only one of the
  five with no reconstruction path.
- *Cannot be verified by the system that wrote it.* This is why the security literature
  exists here and nowhere else: a memory-lifecycle threat model treats poisoning as a
  cross-phase chain — write, persist, propagate, resist cleanup `[C]`
  (arXiv 2604.16548, 2026); optimized triggers achieve >80% attack success at <0.1%
  poison rate with <1% benign degradation `[C]` (arXiv 2407.12784, 2024); and the write
  channels include **compaction-driven writes**, which means your summarizer is an attack
  surface `[C]` (arXiv 2606.04329, 2026). Harm accumulates monotonically with exposure
  length, so short-horizon evals systematically under-report it `[C]`
  (arXiv 2605.17830, 2026).
- *Cannot localize correctness to a record.* Correctness is a property of the state
  trajectory, which is precisely why record-level database operations do not suffice
  `[C]` (arXiv 2605.26252, 2026).

**CONTESTED — is this memory at all?** "Contextual Agentic Memory is a Memo, Not True
Memory" argues current systems are externalized note-taking with no consolidative
process `[C]` (arXiv 2604.27707, 2026); "Storage Is Not Memory" makes an adjacent
retrieval-centred argument `[C]` (arXiv 2605.04897, 2026); the MemOS/MemCube line argues
the opposite, that memory becomes real once it is a scheduled resource spanning plaintext,
activation and parameter tiers `[C]` (arXiv 2507.03724, 2025). Both live. Also contested:
**where the control plane belongs** — an architectural study across thirteen
configurations finds that *where* the LLM sits in the memory pipeline determines which
failure modes are even addressable, with mutation-time placement winning, contradicting
the common assumption that retrieval-time reranking is the leverage point `[C]`
(arXiv 2606.15903, 2026). That is directly load-bearing for a Mnemosyne plug-point
decision.

---

## 8. The property that actually partitions the five

Capacity does not separate them cleanly, and neither does speed. **Reconstructibility
does.** Ask one question of each tier: *if I delete this, where do I get it back from?*

| Tier | Reconstruct from | Cost of loss |
|---|---|---|
| KV cache | the token sequence, exactly | FLOPs. Never information. |
| Recurrent state | the token sequence, exactly | FLOPs. Never information. (The state is lossy *w.r.t.* the tokens — but that loss is a property of the architecture, not of deleting the state.) |
| Retrieval index | the corpus | embedding FLOPs |
| Weights | the training data + the training run | very large, and usually the data is gone |
| **Session store** | **nowhere** | **the information itself** |

Three consequences fall straight out of this table, and they are the reason it is worth
drawing:

1. **The KV cache is not a storage tier, it is a memo table.** Every intuition you have
   about caches — dirty pages, writeback, durability, coherence — is inapplicable, because
   there is nothing to be coherent *with*. Eviction policy here is a compute/quality
   tradeoff, not a data-safety tradeoff. This is exactly why H2O-style eviction `[C]`
   (arXiv 2306.14048, 2023) and attention-sink pinning `[C]` (arXiv 2309.17453, 2023) are
   respectable engineering rather than reckless: the worst case is a worse answer, never
   a lost record.
2. **Only the session store needs the machinery of a real storage system** — durability,
   provenance, integrity, access control, an eviction policy that can be *wrong* in the
   correctness sense. Applying that machinery to the KV cache is over-engineering;
   omitting it from the session store is the bug the 2026 security literature is
   documenting.
3. **The five are not tiers of one hierarchy, despite the OS framing.** A hierarchy
   implies promotion, demotion, and a miss path. Two of the five have no miss path at
   all, and no pair of them has a working promotion path in shipped systems. The
   hierarchy analogy is a *design aspiration* (MemOS's scheduler, `[C]` arXiv 2507.03724)
   and not a description of anything running today. `[A]` high confidence; the cheapest
   falsifier is a shipped system that demotes a KV entry to a cheaper tier and faults it
   back transparently — offload systems come close, but they move bytes without changing
   representation, which is relocation, not demotion.

---

## 9. House vocabulary for Chiron

Per the NAMING RULE, every name must carry information. The word "memory" unqualified
carries none. From here on, in this repo:

| Use | Never use | Because |
|---|---|---|
| `weights` / parametric store | "long-term memory", "internal memory" | collides with the session store |
| `recurrent-state` | "hidden state memory", "compressive memory" | "compressive" is a property, not the thing |
| `kv-cache` | "activation memory", "working memory", "short-term memory" | all three mean something else to the other camp |
| `retrieval-index` | "external memory", "non-parametric memory" | "external" also describes the session store |
| `session-store` | "long-term memory", "agent memory", "persistent memory" | "agent memory" is the term that collides with `kv-cache` in the wild |

Two standing rules. **When citing a source, name which camp's vocabulary it is using** —
"MemOS's *activation memory* (= our `kv-cache`)". And **never write "memory" as a bare
noun in a hypothesis card**; a hypothesis whose subject is ambiguous cannot fail cleanly.

Experiment arms follow the existing convention: policies acting on the `kv-cache` are
`mnemosyne-<policy>` (`mnemosyne-h2o`, `mnemosyne-window`); architecture arms that change
`recurrent-state` size or the SWA/global list are `proteus-<variant>`.

---

## Open questions

Testable at 20M–300M params on one gfx1151, within the measured **≥62 GiB fast tier at
~200 GB/s** and the hard constraint that single tensors ≥32 GiB hang or fault `[M]`
(2026-07-26, `ASSUMPTIONS.md`). No multi-GPU. bf16 numerics still unproven, so the
Hardware Validation Gate precedes all of these.

1. **Is the arithmetic-intensity prediction real on this silicon?** §5 predicts decode
   intensity = `g` FLOP/byte against a measured machine balance of ~105 FLOP/byte. Sweep
   GQA group size `g in {1, 2, 4, 8}` at matched params and check decode throughput tracks
   `g` until it stops. If it does not, the bandwidth model that underpins the whole track
   is wrong on this hardware, and that is worth knowing before anything else is built.
2. **Where is the recall crossover in bytes?** At matched params and matched tokens, at
   what `state_bytes` does a constant-state model match a KV model on MQAR? Zoology gives
   the harness `[C]` (arXiv 2312.04927). The answer in *bytes*, not in architecture names,
   is the capacity-planning number this lab actually needs.
3. **Does state capacity scale linearly with state bytes, or sublinearly?** Same sweep,
   measuring recalled key-value pairs vs state size. Sublinearity would be direct evidence
   for an interference floor rather than a storage limit.
4. **Does HOLA replicate one band down?** A bounded exact KV cache alongside a delta-rule
   compressive state is reported at 340M params / 15B tokens `[C]` (arXiv 2607.02303,
   Jul 2026) — just above our ceiling, single-author, unreplicated. Reproducing at
   ~150M / 2B tokens is the most Mnemosyne-shaped experiment currently available, and a
   failure to replicate is as publishable internally as a success.
5. **Is a quantized KV cache still a pure memo table?** §8 claims the KV cache carries no
   information beyond the tokens. Under 4-bit or 2-bit KV that claim needs a caveat, and
   the caveat is measurable: compare bit-exact recompute against cached decode and report
   logit divergence. If they differ materially, quantized KV is a lossy *information*
   store and every §8 consequence needs restating for it.
6. **Parametric vs context attribution.** Train a ~100M model on synthetic facts, then
   present a conflicting fact in context. Which wins, and can a probe attribute the
   answer to `weights` vs `kv-cache`? This is the attribution instrumentation the
   literature is weakest on, and it needs no scale.
7. **Do compaction policies transfer across the five legs, as the rate-distortion frame
   claims `[C]` (arXiv 2607.08032)?** Cheapest version: score an H2O-style attention-mass
   rule against plain recency for *session-store* consolidation on a synthetic multi-turn
   task. If a KV policy transfers, the unification has teeth; if not, it is a metaphor.

---

## Sources

Every arXiv id below was resolved against the live arXiv API on 2026-07-26.

**Taxonomy and vocabulary**
- arXiv [2607.08032](https://arxiv.org/abs/2607.08032) — *What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction in LLMs and Agents* (Jul 2026). The unification proposal; seven-axis taxonomy across KV eviction, prompt pruning, bounded state, agent consolidation.
- arXiv [2505.00675](https://arxiv.org/abs/2505.00675) — *Rethinking Memory in LLM based Agents: Representations, Operations, and Emerging Topics* (2025). Parametric vs contextual split, six atomic operations.
- arXiv [2507.03724](https://arxiv.org/abs/2507.03724) — *MemOS: A Memory OS for AI System* (2025). The parametric / activation / plaintext trichotomy and the scheduler framing.
- arXiv [2607.02574](https://arxiv.org/abs/2607.02574) — *From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving* (Jun 2026). The serving-side axes: locality, lifetime, ownership, substrate.
- arXiv [2412.19442](https://arxiv.org/abs/2412.19442) — *A Survey on LLM Acceleration based on KV Cache Management* (2024). The canonical token/model/system partition.
- arXiv [2603.07670](https://arxiv.org/abs/2603.07670) — *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers* (Mar 2026).
- arXiv [2602.06052](https://arxiv.org/abs/2602.06052) — *Rethinking Memory Mechanisms of Foundation Agents in the Second Half: A Survey* (Feb 2026). Substrate x cognitive mechanism x subject.
- arXiv [2504.15965](https://arxiv.org/abs/2504.15965) — *From Human Memory to AI Memory* (2025). The cognitive-analogy camp, stated explicitly.
- arXiv [2404.13501](https://arxiv.org/abs/2404.13501) — *A Survey on the Memory Mechanism of LLM based Agents* (2024). Provenance for later taxonomies.
- arXiv [2602.19320](https://arxiv.org/abs/2602.19320) — *Anatomy of Agentic Memory: Taxonomy and Empirical Analysis of Evaluation and System Limitations* (Feb 2026).
- arXiv [2603.04428](https://arxiv.org/abs/2603.04428) — *Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices* (Feb 2026). Cited as evidence of the collision, not for its method.

**Parametric**
- arXiv [2404.05405](https://arxiv.org/abs/2404.05405) — *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws* (2024). ~2 bits/param.
- arXiv [2505.24832](https://arxiv.org/abs/2505.24832) — *How much do language models memorize?* (2025). ~3.6 bits/param, different quantity.
- arXiv [2202.05262](https://arxiv.org/abs/2202.05262) — *Locating and Editing Factual Associations in GPT* (2022). ROME.
- arXiv [2210.07229](https://arxiv.org/abs/2210.07229) — *Mass-Editing Memory in a Transformer* (2022). MEMIT.
- arXiv [2307.12976](https://arxiv.org/abs/2307.12976) — *Evaluating the Ripple Effects of Knowledge Editing* (2023).
- arXiv [2401.07453](https://arxiv.org/abs/2401.07453) — *Model Editing at Scale leads to Gradual and Catastrophic Forgetting* (2024).

**Recurrent state**
- arXiv [2312.00752](https://arxiv.org/abs/2312.00752) — *Mamba* (2023).
- arXiv [2405.21060](https://arxiv.org/abs/2405.21060) — *Transformers are SSMs (SSD)* (2024). State size as one dial.
- arXiv [2312.04927](https://arxiv.org/abs/2312.04927) — *Zoology* (2023). MQAR as the capacity diagnostic.
- arXiv [2402.18668](https://arxiv.org/abs/2402.18668) — *Simple linear attention LMs balance the recall-throughput tradeoff* (2024).
- arXiv [2607.02303](https://arxiv.org/abs/2607.02303) — *A Hippocampus for Linear Attention* (Jul 2026). Compressive state + bounded exact KV, at 340M.
- arXiv [2607.07386](https://arxiv.org/abs/2607.07386) — *Sparse Delta Memory* (Jul 2026). Sparse addressed reads/writes into a large explicit memory.
- arXiv [2506.04761](https://arxiv.org/abs/2506.04761) — *Log-Linear Attention* (2025). Attacks the O(1)-state premise directly.

**KV cache**
- arXiv [1911.02150](https://arxiv.org/abs/1911.02150) — *Fast Transformer Decoding: One Write-Head is All You Need* (2019). Decode is bandwidth-bound.
- arXiv [2305.13245](https://arxiv.org/abs/2305.13245) — *GQA* (2023). The `n_kv` knob.
- arXiv [2405.04434](https://arxiv.org/abs/2405.04434) — *DeepSeek-V2* (2024). MLA, low-rank latent KV.
- arXiv [2306.14048](https://arxiv.org/abs/2306.14048) — *H2O* (2023).
- arXiv [2309.17453](https://arxiv.org/abs/2309.17453) — *StreamingLLM / attention sinks* (2023).
- arXiv [2510.00231](https://arxiv.org/abs/2510.00231) — *The Pitfalls of KV Cache Compression* (2025, rev. 2026). Instruction dropping that LongBench hides.
- arXiv [2603.20397](https://arxiv.org/abs/2603.20397) — *KV Cache Optimization Strategies* (Mar 2026). No single method dominates.
- arXiv [2607.08057](https://arxiv.org/abs/2607.08057) — *System-Aware KV Cache Optimization* (Jul 2026). Temporal / spatial / structural framing.
- arXiv [2309.06180](https://arxiv.org/abs/2309.06180) — *PagedAttention* (2023).
- arXiv [2312.07104](https://arxiv.org/abs/2312.07104) — *SGLang / RadixAttention* (2023).
- arXiv [2407.00079](https://arxiv.org/abs/2407.00079) — *Mooncake* (2024).

**Retrieval**
- arXiv [2005.11401](https://arxiv.org/abs/2005.11401) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020).
- arXiv [2501.01880](https://arxiv.org/abs/2501.01880) — *Long Context vs. RAG for LLMs: An Evaluation and Revisits* (2025).

**Session store**
- arXiv [2310.08560](https://arxiv.org/abs/2310.08560) — *MemGPT* (2023).
- arXiv [2502.12110](https://arxiv.org/abs/2502.12110) — *A-MEM* (2025).
- arXiv [2504.19413](https://arxiv.org/abs/2504.19413) — *Mem0* (2025). Production deployment angle.
- arXiv [2605.26252](https://arxiv.org/abs/2605.26252) — *Is Agent Memory a Database?* (May 2026). Correctness as a property of the state trajectory.
- arXiv [2604.27707](https://arxiv.org/abs/2604.27707) — *Contextual Agentic Memory is a Memo, Not True Memory* (Apr 2026).
- arXiv [2605.04897](https://arxiv.org/abs/2605.04897) — *Storage Is Not Memory* (May 2026).
- arXiv [2606.24775](https://arxiv.org/abs/2606.24775) — *Are We Ready For An Agent-Native Memory System?* (Jun 2026).
- arXiv [2606.15903](https://arxiv.org/abs/2606.15903) — *Control-Plane Placement Shapes Forgetting* (Jun 2026).
- arXiv [2604.16548](https://arxiv.org/abs/2604.16548) — *A Survey on Long-Term Memory Security in LLM Agents* (Apr 2026, rev. Jun 2026).
- arXiv [2606.04329](https://arxiv.org/abs/2606.04329) — *From Untrusted Input to Trusted Memory* (Jun 2026). Write-channel taxonomy; compaction as attack surface.
- arXiv [2407.12784](https://arxiv.org/abs/2407.12784) — *AgentPoison* (2024).
- arXiv [2605.17830](https://arxiv.org/abs/2605.17830) — *Remembering More, Risking More* (May 2026).
- arXiv [2605.23296](https://arxiv.org/abs/2605.23296) — *Parallel Context Compaction for Long-Horizon LLM Agent Serving* (May 2026).
- arXiv [2507.05257](https://arxiv.org/abs/2507.05257) — *Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions* (2025). MemoryAgentBench.

**Internal — measured on this lab's hardware, and read from this lab's reference clones**
- `ASSUMPTIONS.md`: `gpu-fast-tier-size`, `kv-per-token-laguna`, `laguna-heads-uniform`, `hipblaslt-config`, `gemm-throughput-below-reference`, `large-tensor-fault-32gib` (all `[M]`, 2026-07-26).
- `notebook/uma-carveout-controls-fast-tier.md` — bandwidth tier sweep, ≥62 GiB at ~200 GB/s.
- `research/reference/CODE_MAP.md` — verified `file:line` pointers used above: Laguna `modeling_laguna.py:365`; vLLM `single_type_kv_cache_manager.py:97`, `block_pool.py:647/679/702/719`, `kv_cache_utils.py:596`; SGLang `radix_cache.py:355/565/676`; Mamba-2 `mamba2.py:317/352`, `ssd_combined.py:375`; Gated DeltaNet `naive.py:54/56/58`, `fused_recurrent.py:136/153`; Mooncake `master_service.cpp:6382`.
