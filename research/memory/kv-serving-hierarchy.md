---
title: The serving layer as a memory hierarchy — and the four places the analogy breaks
version: 1.0.0
date: 2026-07-26
track: research/memory (note 4 of 10)
---

# The serving layer as a memory hierarchy

This note settles three things. First, that the KV cache genuinely *is* a storage
hierarchy problem — the arithmetic that makes decode bandwidth-bound is written out
below from measured numbers on our own box, and every serving system in the
literature is a different answer to that one inequality. Second, that four of the
five instincts a storage engineer imports are wrong here in specific, nameable ways:
there is no fault path, eviction granularity is the request rather than the page,
the cache key is a chain rather than a content hash, and discarding data is legal
because a KV block is recomputable — a freedom no real storage tier has. Third, that
the Strix Halo box this lab runs on collapses the exact tier boundary (GPU-local vs
host DRAM across PCIe) that the entire offload and CXL literature is built around,
which makes most of that literature's *economics* untransferable to us while making
its *policy questions* unusually cheap for us to test.

---

## 1. The one inequality

Everything downstream is a response to a ratio. Symbols first, all of them:

| Symbol | Meaning |
|---|---|
| `L` | number of transformer layers |
| `H_q` | query heads per layer |
| `H_kv` | key/value heads per layer (with GQA, `H_kv ≤ H_q`) |
| `d` | head dimension — channels per head |
| `S` | tokens already in the cache (context length) |
| `b` | bytes per stored KV element (2 for bf16/fp16, 1 for fp8) |
| `P` | parameters activated per token |
| `F` | achieved FLOP/s |
| `BW` | achieved bytes/s of the memory tier holding the cache |

**KV bytes per token.** Each layer stores one key vector and one value vector per
KV head, hence the leading 2:

```
k = 2 · L · H_kv · d · b        [bytes per token]
```

For the reference model (Laguna S 2.1: `L`=48, `H_kv`=8, `d`=128, bf16) that is
`2·48·8·128·2 = 196,608 B = 192 KiB per token` `[M]` (ASSUMPTIONS `kv-per-token-laguna`,
read from the fetched config at `b0a9fd7c850e`, 2026-07-26). Caveat carried from that
same row: attention head counts vary per layer in Laguna
(`modeling_laguna.py:343` takes `num_heads` from `config.num_attention_heads_per_layer`),
so 192 KiB is an upper bound, not a per-layer-exact figure `[M]`
(ASSUMPTIONS `laguna-heads-uniform`, **refuted**).

**Decode arithmetic intensity.** To emit one token, attention must read the entire
cache and does roughly `4 · L · H_q · d · S` FLOPs against it (two matmuls, QK^T and
AV, each counting a multiply-add as 2 FLOPs). Divide FLOPs by bytes and `L`, `d`, `S`
all cancel:

```
I = (4 · L · H_q · d · S) / (2 · L · H_kv · d · S · b)  =  2 · H_q / (H_kv · b)
```

Three worked cases, all at bf16 (`b`=2):

- MHA (`H_q = H_kv`): **I = 1 FLOP/byte**
- Laguna-shaped GQA (`H_q`=48, `H_kv`=8): **I = 6 FLOP/byte**
- MQA (`H_kv`=1, `H_q`=48) `[C]` 1911.02150: **I = 48 FLOP/byte**

Now the machine. Our measured GEMM peak is **20.9 TFLOPS bf16 at 8192³** and our
measured device-to-device bandwidth is **~200 GB/s flat out to a 62 GiB footprint**
`[M]` (`scripts/benchmark_gemm.py` and `notebook/uma-carveout-controls-fast-tier.md`,
both 2026-07-26, single run each — anecdotes by the house standard, but with ~2x effect
sizes and sharp boundaries). Machine balance is therefore

```
β = F / BW = 20.9e12 / 200e9 ≈ 104.5 FLOP/byte
```

Decode attention at `I` = 6 sits **~17x below machine balance**. Even MQA at bf16 sits
below it. This is the whole reason the serving literature exists, and it is why
`[C]` 1911.02150 (2019) is still the correct first read: decode is not a compute
problem, it is a streaming problem, and every technique below is a way of moving fewer
bytes or moving them from somewhere closer.

**The batching asymmetry — this is the part that surprises systems people.** Weight
reads amortize across a batch: with batch `N`, per-token weight traffic is `P·b_w/N`.
KV reads do *not* amortize, because each sequence owns its own cache. Setting the two
equal gives the crossover:

```
S · N  =  P · b_w / k
```

Arithmetic (not a measurement) for a hypothetical 300M dense ablation model at our
scale — `P`=3e8, `b_w`=2, `L`=24, `H_kv`=4, `d`=64, `b`=2, so `k` = 24 KiB/token:
`S·N = 600e6 / 24576 ≈ 24,400 token-slots`. At batch 32 the crossover is at
**S ≈ 760 tokens**. Past that, the decode step is dominated by KV streaming and
increasing batch size stops helping. A storage engineer will recognise the shape: this
is a per-connection working set, not a shared one, so the usual "add more clients to
amortize the seek" reflex is exactly backwards.

The single exception — and it is the reason half this note exists — is a KV block that
is *shared* between sequences. A shared prefix block is read once and used by many
queries, which is the only mechanism in the stack that actually raises attention's
arithmetic intensity rather than just reducing bytes.

---

## 2. PagedAttention: a page table, until it isn't

`[C]` 2309.06180 (Sep 2023) is where the OS vocabulary enters. Before it, a serving
system reserved `max_seq_len` of contiguous KV per sequence. At Laguna's 192 KiB/token
that is **24 GiB reserved per sequence** for a 128k window, almost all of it unused.
Paging replaces the reservation with fixed-size blocks of `B` tokens, so waste is
bounded at under one block per sequence: `(B−1) · k` = 15 × 192 KiB = **2.88 MiB** at
vLLM's default `B`=16. Twenty-four gigabytes to three megabytes. That is the entire
value proposition, and it is the same argument as `malloc` over static partitioning.

The structure is where you expect it. Scheduler-side, `req_to_blocks` is a per-sequence
page table — list index is the logical KV block, `KVCacheBlock.block_id` is the physical
frame (`memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:97`) `[C]` CODE_MAP.
Worker-side it is flattened into a dense `[max_num_reqs, max_num_blocks_per_req]` int32
matrix and memcpy'd to the GPU each step (`vllm/v1/worker/block_table.py:81`), and the
attention kernel performs the translation itself with one load per KV tile:
`block_tables_ptr + seq_idx*stride + seq_offset//BLOCK_SIZE`
(`vllm/v1/attention/ops/triton_unified_attention.py:424`). A software page walk, once
per tile per head per layer, with no MMU and no TLB.

Note the metadata cost while you are here. At 256 concurrent requests and a 128k window
with `B`=16, that dense table is 256 × 8192 × 4 B = **8 MiB copied to the device every
decode step** (arithmetic from the verified structure above, not a measurement). Against
gigabytes of KV that is cheap, but "metadata-to-data ratio in shared stores" is the
first of seven measurement gaps the 2026 survey names as unmeasured across the whole
field `[C]` 2607.02574 (Jun 2026).

### Break 1 — there is no page fault

Nothing is demand-paged. `allocate_slots` returning `None` is not a fault to be
serviced; it is an admission rejection, and the scheduler preempts the **entire
request** (`block_pool.py:647`, `get_new_blocks`) `[C]` CODE_MAP. So eviction
granularity is the sequence, not the page. Every capacity-planning intuition that
depends on graceful degradation under pressure — page out the cold quarter of a working
set and keep running — is unavailable. The system either holds the whole sequence or
drops it and starts over.

### Break 2 — the free list *is* the victim cache

`FreeKVCacheBlockQueue` (`kv_cache_utils.py:184`) is an intrusive doubly-linked list
threaded through the block objects, and blocks sit on it with their contents and hashes
intact so a later prefix hit can resurrect one via `touch()` (`block_pool.py:702`)
`[C]` CODE_MAP. Freeing is not evicting: `free_blocks` only decrements the refcount and
leaves the hash-table entry live, and actual eviction happens lazily inside
`get_new_blocks → _maybe_evict_cached_block` at the moment of reallocation
(`block_pool.py:679`). Consequence for observability, which matters to this lab: "blocks
in use" and "blocks still matchable for a hit" are two different numbers, and neither is
"free memory". Any dashboard that reports one of the three as cache occupancy is lying.

### Break 3 — the page size is denominated in tokens, and a hybrid model has several

`page_size_bytes` differs per layer type in `kv_cache_interface.py`, so a hybrid
SWA/global model runs several independent block tables with different frame geometry
over one physical pool `[C]` CODE_MAP. No single-address-space page table has to model
that. It gets worse at lookup: `get_cached_block` packs `(block_hash, group_id)` into
one key and **returns `None` if any KV cache group misses** (`block_pool.py:198`), so a
hybrid model's prefix hit rate is bounded by its least-cacheable group. For a lab whose
reference model is 12 global + 36 sliding layers in a strict GSSS pattern `[M]`
(ASSUMPTIONS `reference-model`), that is not a footnote — it is a first-order property
of the arm.

### Break 4 — deduplication is deliberately refused

vLLM does not check whether an identical block already exists when caching a full block;
two sequences that independently produce identical KV keep two physical copies forever.
The stated reason (`block_pool.py:47-51` docstring) is that collapsing them would change
an already-issued `block_id`, and block-table rows must remain strictly append-only
`[C]` CODE_MAP. This is the exact opposite of KSM/page-dedup and of the paper's own
copy-on-write framing: sharing happens only at lookup time, before allocation, never
retroactively. An append-only invariant is being paid for in wasted physical memory.

### Contested: is paging even the right primitive?

`[C]` 2405.04437 (vAttention, ASPLOS'25) argues no — CUDA VMM APIs decouple virtual from
physical allocation, giving the same defragmentation while keeping the KV cache
*virtually contiguous*, so stock attention kernels need no rewrite. `[C]` 2607.02574
(Jun 2026) still lists this as unresolved. Note the meta-point: "of course you page" is
precisely the systems instinct under dispute, and the reason it is disputable is that
GPU attention kernels, unlike CPU loads, must be *taught* about non-contiguity by hand.
Do not treat vLLM's dominance as evidence; it is evidence about ecosystem gravity.

---

## 3. Cross-request prefix reuse: the only lever on intensity

Two shipped designs, same goal, different data structure.

**vLLM — a chained hash.** A block's key is `hash(parent_block_hash, token_ids,
extra_keys)` (`kv_cache_utils.py:596`), folding the whole prefix in `[C]` CODE_MAP. The
match loop walks from token 0 and **breaks at the first miss**, because a later hit is
impossible by construction (`single_type_kv_cache_manager.py:658`). Two consequences a
cache engineer should feel immediately: the same 16 tokens at a different offset are a
different key, and one changed token at position 0 invalidates every downstream hash.
There is no associative match and no middle-of-sequence match.

**SGLang — a radix tree.** `TreeNode` owns a *run* of KV slots, and `_split_node`
cleaves that run when a new sequence diverges mid-node — the same move as splitting a
filesystem extent on a partial overwrite (`radix_cache.py:217`, `:676`) `[C]` CODE_MAP,
`[C]` 2312.07104. Lookup is O(depth) rather than O(prefix length)
(`_match_prefix_helper`, `:650`), and the entire replacement-policy surface is one
`get_priority(node)` function — LRU, LFU, FIFO, MRU, segmented-LRU all live in a 65-line
file (`evict_policy.py:16`). That is the cleanest pluggable-policy interface in the
reference library and is worth copying wholesale into Mnemosyne.

### Break 5 — eviction is topologically constrained, not recency-ordered

SGLang's `evict()` only ever considers **leaves**, drawn from an incrementally
maintained `evictable_leaves` set (`radix_cache.py:565`, `_update_leaf_status` at
`:790`), peeling the frontier inward. A hot child therefore keeps a cold parent resident
indefinitely, and `inc_lock_ref` walks to the root, letting one in-flight request pin an
arbitrarily deep chain `[C]` CODE_MAP. This is a pinned dentry chain, not an LRU list,
and no amount of LRU intuition predicts its behaviour under a workload with one long-
lived deep prefix. Related trap: `match_prefix` is a **mutating read** — it splits nodes
and clones tensors — so "lookup" is neither cheap nor concurrency-safe.

### Break 6 — a hit removes compute, not I/O, and the working set is not the model's

The thing a prefix hit saves is a forward pass, not a disk read. And the working set is
the union of live conversation prefixes, which means hit rate is governed by **request
routing** far more than by cache size. That has turned prefix affinity into a scheduling
problem in its own right over the last six months: `[C]` 2602.06502 (DualMap, Feb 2026)
frames cache affinity against load balancing as a direct trade-off, since scattering
requests reduces hit rate and raises recompute; `[C]` 2602.11688 (Feb 2026) optimises
prefix reuse against cross-region network latency; `[C]` 2605.08581 (PRISM, May 2026)
co-designs the scheduler with a demand-aware radix tree to protect high-value shared
prefixes; `[C]` 2607.02525 (PEEK, 2026) makes cache management queue-informed. `[C]`
2312.07104's original claim — that hit rate stops being an implementation detail and
becomes a scheduling objective — is now the mainstream position.

One more sharp edge, because it will show up in any hit-rate measurement you run: a 100%
prefix match never skips 100% of the work. `get_computed_blocks` caps the hit at
`num_tokens − 1` (you need a forward pass to get logits) and then floors to block
alignment, so an exact-duplicate prompt still recomputes a full trailing block
`[C]` CODE_MAP. Report hit rate in tokens-skipped, not blocks-matched, or the number
flatters itself.

---

## 4. Prefill and decode are two different machines

The phase asymmetry falls straight out of §1. Prefill processes `S` tokens at once:
weight reads amortize over `S`, attention is a dense matmul, and intensity is high —
compute-bound. Decode processes one token: nothing amortizes, and intensity is
`2·H_q/(H_kv·b)` — bandwidth-bound. They also have different SLOs, TTFT for prefill and
TPOT for decode, and co-locating them means a long prefill stalls every decode in the
batch.

`[C]` 2401.09670 (DistServe) and `[C]` 2311.18677 (Splitwise) answer by putting the two
phases on different GPUs, which converts the interference problem into a **KV transfer**
problem: the prefill worker's cache must reach the decode worker. `[C]` 2403.02310
(Sarathi-Serve, OSDI'24) answers by chunking prefill into stall-free slices inside one
pool, capturing much of the benefit with no transfer at all.

**Contested, and actively so.** `[C]` 2508.01989 (Aug 2025) argues explicitly for
unifying both — differentiated-capability instances, prefill-heavy and decode-heavy,
chosen per SLO combination — rather than picking. `[C]` 2607.02043 (Jul 2026) adds
load-aware prefill deflection to disaggregated deployments. `[C]` 2606.29708 (Jun 2026)
maps the heterogeneous-serving design space rather than declaring a winner. The 2026
survey `[C]` 2607.02574 lists disaggregation-vs-chunked-prefill as unresolved. Present
it as unresolved.

**Directly relevant constraint for us:** distributed collectives are incomplete on
gfx1151, so we are single-device `[C]`/`[M]` (ASSUMPTIONS `single-device-only`). We
cannot run a disaggregated arm. We *can* measure the phase asymmetry itself and we *can*
run chunked prefill, because both are single-pool.

---

## 5. Offload tiering, and the inequality that decides it

The offload question is: for a reusable prefix of `S_p` tokens, is it cheaper to load
its KV from a slower tier or to recompute it? Write both sides.

```
t_load       = k · S_p / BW              k = KV bytes/token
t_recompute  = 2 · P · S_p / F           (forward pass ≈ 2 FLOPs per param per token)
```

Loading wins when `k·S_p/BW < 2·P·S_p/F`. **`S_p` cancels**, which is the interesting
part — to first order the decision is independent of how long the prefix is, and depends
only on model shape and tier bandwidth:

```
BW  >  F · k / (2 · P)
```

Arithmetic with our measured `F` = 20.9 TFLOPS `[M]`:

- Laguna-shaped (`k` = 192 KiB, `P` = 8.5e9 active): threshold **≈ 0.24 GB/s**
- A 300M dense ablation model (`k` = 24 KiB, `P` = 3e8): threshold **≈ 0.86 GB/s**

Any real storage tier clears both by two orders of magnitude, which is why the field
converged on "store and reload, do not recompute." Independent confirmation of the
direction: `[C]` 2601.19910 reports recompute at 28.5× the cost of a CPU round-trip for
a 4,096-token context. (Note the `S_p²` attention term in prefill, omitted above, only
makes recompute worse at long context.)

**But the same paper is the strongest argument against naive offload.** `[C]` 2601.19910
characterises PCIe-bound KV offload and finds **99% of latency spent in transfers, with
GPUs drawing only 28% of rated TDP**, and derives `κ_crit`, the cached-to-prefill token
ratio past which execution is memory-bound — exceeded by typical workloads by orders of
magnitude. So offload buys capacity and loses the machine. That is a throughput
casualty, not a latency one, and it is invisible in a latency-only benchmark.

The engineering response is a layer beneath the engine: `[C]` 2510.09665 (LMCache, 2025)
implements offload to CPU/disk/object store plus cross-engine transfer for PD
disaggregation. Prediction and prefetch are the 2026 direction — `[C]` 2604.26968
(predictive multi-tier KV management, Apr 2026), `[C]` 2603.27138 (ScoutAttention,
layer-ahead CPU pre-computation, Mar 2026), `[C]` 2605.24022 (adaptive KV reuse, May
2026).

### Break 7 — the tiers on our machine are not the tiers in these papers

Every paper above assumes a ladder: GPU HBM at TB/s, host DRAM at ~100 GB/s, PCIe at
tens of GB/s between them. Strix Halo has one physical pool. Our measured curve is
**flat at ~200 GB/s from 2 GiB to 62 GiB of footprint** with the BIOS UMA carve-out at
96 GB, where at the 16 GiB default it collapsed to 61 GB/s past a 30 GiB footprint
`[M]` (`notebook/uma-carveout-controls-fast-tier.md`, 2026-07-26). So:

- The tier boundary is real, is **~2x**, and is set by a **BIOS setting**, not by a bus.
- "Offload to CPU DRAM" is close to a no-op here: same physical DRAM, so it buys no
  capacity (the pool is shared with the host, which dropped to 31.6 GB visible RAM) and
  plausibly no bandwidth. `[A]` medium confidence that host-side access runs at
  comparable bandwidth; **cheapest test:** measure host-to-device copy and CPU-only
  memcpy bandwidth on the same box and compare against the 200 GB/s device-to-device
  figure.
- Therefore results about offload *economics* do not transfer to us. Results about
  offload *policy* — what to keep hot, when to prefetch, how to score a block — become
  unusually cheap to test, because we can create a genuine two-tier experiment with a
  driver setting and no interconnect in the way.

Standing hazard on any such experiment: single tensors ≥32 GiB hang or fault on this
stack, and the hang presents at 0 CPU with no error `[M]` (ASSUMPTIONS
`large-tensor-fault-32gib`, **refuted**). Keep individual KV buffers under 32 GiB.

---

## 6. Mooncake: the KV cache as a distributed store

`[C]` 2407.00079 (Mooncake) is what the cache becomes once it outlives a request. Read
the implementation as two machines glued by a lease `[C]` CODE_MAP:

- **Data plane.** Three verbs: `registerLocalMemory` (pin and advertise),
  `openSegment` (handle on a peer's registered region), `submitTransfer` (batch of
  READ/WRITE against segment id + byte offset) — `transfer_engine.h:117`. This is
  `pread`/`pwrite` against someone else's address space, with the target CPU uninvolved.
  It knows nothing about keys.
- **Control plane.** A separate metadata master owns key → (segment, offset, replica
  type). A write is a two-phase commit: `PutStart` allocates and returns addresses, the
  client RDMA-writes the bytes itself, `PutEnd` publishes (`master_service.h:382`).
- **Tiering.** `SelectBestReplica` is a fixed preference ladder — local DRAM > local
  NVMe-over-fabric > remote DRAM > remote NOF > local disk > disk
  (`replica_selection.h:122`). Note: a fixed order, **not** a latency/cost model.
- **Placement.** `FreeRatioFirstAllocationStrategy` samples 6N candidate segments, sorts
  by free ratio, takes the top N (`allocation_strategy.h:402`) — power-of-k-choices.
- **Writeback and promotion.** `BatchOffload` (`storage_backend.h:354`) is DRAM→SSD under
  high/low watermarks; `TryPushPromotionQueue` (`master_service.cpp:5211`) gates
  disk→DRAM behind a count-min-sketch frequency threshold, TinyLFU-style, so a single
  cold hit never pollutes the fast tier.

### Break 8 — the working set is defined by leases, not recency

`BatchEvict` (`master_service.cpp:6382`) only considers objects whose **lease has
expired**, then partially sorts those by lease deadline with `nth_element` `[C]`
CODE_MAP. `Get` renews the lease. So the "hot" signal is a TTL renewal, not a touch bit,
and this is closer to leased pages in a distributed filesystem than to an LRU chain.
Two details that cut against the paper's framing: `eviction_strategy.h` ships textbook
`LRUEvictionStrategy` and `FIFOEvictionStrategy` classes that **nothing in the production
path uses**, and a disk-tier hit does not automatically fill DRAM — promotion is a
throttled background task a single access is designed to lose.

### Break 9 — the freedom no storage tier has

Evicting a KV block is never data loss. It is a recompute. That is why
`offload_force_evict` can throw bytes away rather than block on writeback `[C]` CODE_MAP.
Internalise this: **every durability guarantee you have ever engineered is optional
here**, and the correctness argument for any Mnemosyne policy is not "did we lose data"
but "did we spend more recomputing than we saved." Conversely, this is also the trap —
because loss is legal, systems ship without loss accounting, and nobody measures how
often a discard was later regretted. That is precisely the attribution gap this lab
claims as its edge.

**Contested: where ownership lives.** Coordinator-based stores (Mooncake) simplify
policy but bottleneck; peer protocols distribute load and add complexity. `[C]`
2607.02574 names ownership as the axis that most predicts a system's design and the
least settled. It also names an **empty design point** — per-session KV lifetime —
despite chat and agent workloads obviously wanting session-scoped retention.

---

## 7. CXL-pooled KV, and why our box is an accidental preview

CXL replaces "copy the bytes to the consumer" with "the consumer loads them" —
byte-addressable, cache-line granularity, shared across hosts. `[C]` 2512.18194 (TraCT,
Dec 2025) is the anchor: CXL shared memory used as both a network-free KV transfer
substrate for PD disaggregation and a rack-scale prefix-aware cache, reporting up to
2.6× lower average TTFT, 3.0× lower P99, and 1.9× higher peak goodput than RDMA
baselines (vendor/author-reported, single system, not independently replicated).

Since then this has become a subfield, all of it post-dating most readers' priors:
`[C]` 2512.11920 (CXL-SpecKV — FPGA-side speculative KV prefetch and compression),
`[C]` 2606.19746 (SAC — fetch only the top-k KV entries on demand for sparse-attention
models, exploiting cache-line granularity), `[C]` 2606.12556 (ITME — tiered expansion
with disaggregated CXL-hybrid memories), `[C]` 2511.00321 (processing-near-memory for
1M-token inference), and `[C]` 2607.18141 (HyMCache, Jul 2026 — multi-turn serving over
CXL-hybrid memory).

The genuinely new capability is **granularity**. RDMA and PCIe move blocks; CXL moves
cache lines. That is what makes SAC's "fetch only the needed top-k entries" coherent at
all — under a block-transfer substrate, sparse access is dominated by read
amplification. If sparse attention and dynamic KV selection are the direction of travel
(and the compression track says they are), then substrate granularity stops being an
implementation detail and becomes an architectural constraint on which policies are
even expressible.

**Contested.** Load/store sharing is not free: non-coherent shared memory forces you to
solve publication ordering, visibility, and consistency by hand — which `[C]` 2512.18194
is more instructive about than about its speedups. And hardware availability means every
number in this subfield comes from a small number of testbeds.

### Where this lands for us

Strix Halo is a one-package, zero-hop version of the same idea: CPU and GPU share one
byte-addressable pool with no interconnect between them, which is the property CXL is
being built to approximate across a rack. We get the *topology* for free and lose the
*capacity* argument entirely (the pool is 128 GB total and shared with Windows). The
honest framing for any arm we run on this: it is a natural experiment in
collapsed-hierarchy KV management, and it is a **confound** for any result meant to
generalise to an HBM/DRAM/PCIe machine. Say which one you are claiming, in the
pre-registration, before the run.

---

## 8. The breaks, consolidated

| Systems instinct | What actually happens | Where to see it |
|---|---|---|
| A miss triggers a fault and a fill | No fault path exists. Miss = recompute, or admission rejection | `block_pool.py:647`; llama.cpp iSWA has no promote/demote at all |
| Eviction granularity is the page | It is the **request** (vLLM preemption) or the **leaf subtree** (SGLang) | `radix_cache.py:565` |
| Free list ≠ cache | They are the same intrusive list; freeing is not evicting | `kv_cache_utils.py:184`, `block_pool.py:679` |
| Cache key is content-addressed | It is a **chain** over the whole prefix; position-dependent, prefix-ordered only | `kv_cache_utils.py:596` |
| Identical pages get deduped | Deliberately refused to keep block tables append-only | `block_pool.py:47-51` |
| Hot data gets promoted | Promotion is frequency-gated and a single hit is designed to lose | `master_service.cpp:5211` |
| Recency drives eviction | Leases drive it (Mooncake); leaf topology drives it (SGLang) | `master_service.cpp:6382` |
| Losing data is a bug | Losing data is a *cost*. Discard is always legal | `offload_force_evict` |
| A read is read-only | `match_prefix` splits nodes and clones tensors | `radix_cache.py:355` |
| Tiers are interchangeable copies | Laguna's full and SWA tiers use different RoPE θ — not substitutable | `laguna.cpp:184` |

The last row deserves a sentence, because it is the one that generalises to
architecture rather than to serving. In llama.cpp's Laguna, the two KV caches are
genuinely two tiers by size — full-size for the 12 global layers, `n_swa + n_ubatch` for
the 36 sliding ones (`llama-kv-cache-iswa.cpp:73`) `[C]` CODE_MAP — but there is no
promotion, no demotion, and no miss path, because `is_masked_swa` makes out-of-window
tokens *architecturally unreadable*. Discarding them is lossless rather than a gamble.
And you cannot "just widen the window" to test long context, because SWA layers apply
plain RoPE at θ=10000 over all 128 head dims while full layers apply YaRN over 64 dims
at θ=500000. The tiers are not the same data in two places; they are different data
under different positional encodings.

---

## 9. What this implies for Mnemosyne

Three design consequences, stated as consequences rather than decisions (ADRs go in
`docs/adr/`):

1. **The pluggable surface is `get_priority(node)`.** SGLang has already demonstrated
   that LRU/LFU/FIFO/MRU/segmented-LRU fit behind one scoring function
   (`evict_policy.py:16`). A policy interface that cannot also express lease-expiry
   (Mooncake) and refcount-pinning (vLLM `touch`) is under-specified.
2. **Instrument the discard, not just the hit.** Because discard is legal, no shipped
   system accounts for regretted evictions. A counter of "blocks discarded that were
   subsequently recomputed, and the token-cost of doing so" is cheap, is absent from the
   literature, and is directly the attribution instrumentation this lab claims as its
   contribution. It also maps onto the survey's named gap list `[C]` 2607.02574.
3. **Report hit rate three ways or not at all.** Blocks matched, tokens skipped, and
   forward-pass FLOPs avoided diverge by construction (the `num_tokens − 1` cap and block
   alignment guarantee it). One number here is an anecdote dressed as a metric.

---

## Open questions

Testable on our hardware: single gfx1151 device, 20M–300M params, **~62 GiB measured
fast tier at ~200 GB/s**, no working multi-GPU, individual buffers must stay under
32 GiB.

1. **Does the measured 200 GB/s device-to-device figure hold host-to-device and
   CPU-side?** If yes, "offload to CPU" is bandwidth-neutral on this platform and the
   entire offload literature's cost model collapses to a capacity-accounting question.
   Cheapest test: three-way bandwidth sweep (D2D, H2D, CPU memcpy) at matched footprints.
   Directly resolves the `[A]` in §5.
2. **Where is the upper edge of the fast tier?** The 62 GiB figure is a floor — the sweep
   hit the ≥32 GiB single-buffer fault, not a bandwidth knee `[M]`. Re-run with buffers
   capped at 31 GiB and a footprint sweep to 90 GiB.
3. **Measure the decode roofline directly and check `I = 2·H_q/(H_kv·b)`.** Sweep
   `H_kv` ∈ {1, 2, 4, H_q} at fixed `H_q` on a 300M model and plot achieved tok/s against
   predicted intensity. This validates the note's central arithmetic on our own
   instrument, and it is the prerequisite for trusting any later KV-budget claim.
4. **Does the batching-crossover arithmetic (`S·N ≈ P·b_w/k`) hold?** Predicted ≈760
   tokens at batch 32 for a 300M model. Measured departure would indicate the harness,
   not the theory — and per house rule, if it looks too good, suspect the harness.
5. **Prefix-reuse hit rate under a hybrid model's intersection rule.** vLLM requires all
   KV cache groups to hit. Build a 3:1 SWA/global toy arm and measure how much hit rate
   the sliding group costs versus an all-global control at matched params. As far as this
   note found, unmeasured in public.
6. **Regretted-eviction accounting.** Instrument a policy to count discards later
   recomputed, and report the token cost. Run H2O/SnapKV/window policies against it. This
   is an attribution metric, not an outcome metric, and it is the cheapest genuinely
   novel measurement available to us.
7. **Chunked prefill on a single pool.** Sarathi-Serve's stall-free scheduling needs no
   second device. Measure TTFT/TPOT trade-off curves at our scale and check whether the
   phase asymmetry is even large enough at 300M to be worth scheduling around.
8. **Does the 32 GiB fault bound KV experiments in practice?** A 62 GiB cache built from
   ≤31 GiB shards is fine; a single contiguous KV tensor is not. Determine whether paged
   allocation is *forced* on us by a driver bug rather than chosen — which would be an
   amusing empirical vindication of PagedAttention over vAttention on this platform
   specifically.

---

## Sources

Code pointers are from `research/reference/CODE_MAP.md`, whose every `file:line` is
machine-verified against the pinned revisions in `PROVENANCE.md`. Paper ids are from
`research/reference/papers/anchors.bib` (resolved against the live arXiv API) or were
resolved against the arXiv API during the writing of this note (2026-07-26).

**Serving architecture — foundations**
- `[C]` arXiv 2309.06180 — *Efficient Memory Management for Large Language Model Serving with PagedAttention* (Sep 2023). Block tables, internal/external fragmentation, CoW sharing.
- `[C]` arXiv 2405.04437 — *vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention* (2024, ASPLOS'25). The contiguous-virtual-memory counter-argument.
- `[C]` arXiv 2312.07104 — *SGLang: Efficient Execution of Structured Language Model Programs* (Dec 2023). RadixAttention.
- `[C]` arXiv 1911.02150 — *Fast Transformer Decoding: One Write-Head is All You Need* (2019). Why decode is bandwidth-bound.
- `[C]` arXiv 2305.13245 — *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023). The `H_q → H_kv` knob.
- `[C]` arXiv 2405.04434 — *DeepSeek-V2* (2024). MLA, the low-rank alternative to shrinking `H_kv`.

**Phase disaggregation**
- `[C]` arXiv 2401.09670 — *DistServe* (2024). TTFT/TPOT as separate SLOs; the KV-transfer cost model.
- `[C]` arXiv 2311.18677 — *Splitwise* (2023). Per-phase hardware provisioning.
- `[C]` arXiv 2403.02310 — *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (Mar 2024). Chunked prefill, single pool.
- `[C]` arXiv 2508.01989 — *Prefill-Decode Aggregation or Disaggregation? Unifying Both for Goodput-Optimized LLM Serving* (Aug 2025).
- `[C]` arXiv 2607.02043 — *Towards Load-Aware Prefill Deflection for Disaggregated LLM Serving* (Jul 2026).
- `[C]` arXiv 2606.29708 — *Demystifying the Design Space and Best Practices for Heterogeneous LLM Inference and Serving* (Jun 2026).

**Prefix reuse as a scheduling problem**
- `[C]` arXiv 2602.06502 — *DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving* (Feb 2026).
- `[C]` arXiv 2602.11688 — *GORGO: Online Tuning for Cross-Region Network-Aware LLM Serving* (Feb 2026). Note: title has drifted from the KV-reuse phrasing in earlier listings; cite by id.
- `[C]` arXiv 2605.08581 — *PRISM: Fast Online LLM Serving via Scheduling-Memory Co-design* (May 2026).
- `[C]` arXiv 2607.02525 — *PEEK: Predictive Queue-Informed KV Cache Management for LLM Serving* (2026).

**Offload tiering**
- `[C]` arXiv 2601.19910 — *Understanding Bottlenecks for Efficiently Serving LLM Inference With KV Offloading*. 99% of latency in transfers; GPUs at 28% of rated TDP; the `κ_crit` cached-to-prefill ratio; recompute 28.5× a CPU round-trip at 4,096 tokens.
- `[C]` arXiv 2510.09665 — *LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference* (2025).
- `[C]` arXiv 2604.26968 — *Predictive Multi-Tier Memory Management for KV Cache in Large-Scale GPU Inference* (Apr 2026).
- `[C]` arXiv 2603.27138 — *ScoutAttention: Efficient KV Cache Offloading via Layer-Ahead CPU Pre-computation for LLM Inference* (Mar 2026).
- `[C]` arXiv 2605.24022 — *Adaptive KV Cache Reuse for Fast Long-Context LLM Serving* (May 2026).

**KV as a distributed store**
- `[C]` arXiv 2407.00079 — *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving* (2024).

**CXL-pooled KV**
- `[C]` arXiv 2512.18194 — *TraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale* (Dec 2025). Up to 2.6× lower avg TTFT, 3.0× lower P99, 1.9× higher peak goodput vs RDMA baselines (author-reported).
- `[C]` arXiv 2512.11920 — *CXL-SpecKV: A Disaggregated FPGA Speculative KV-Cache for Datacenter LLM Serving* (Dec 2025).
- `[C]` arXiv 2606.19746 — *SAC: Disaggregated KV Cache System for Sparse Attention LLMs with CXL* (Jun 2026).
- `[C]` arXiv 2606.12556 — *ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories* (Jun 2026).
- `[C]` arXiv 2511.00321 — *Scalable Processing-Near-Memory for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management Beyond GPU Limits* (Oct/Nov 2025).
- `[C]` arXiv 2607.18141 — *HyMCache: A KV Cache Framework for Multi-Turn LLM Serving with CXL-Hybrid Memory* (Jul 2026).

**Surveys that frame the field as a hierarchy**
- `[C]` arXiv 2607.02574 — *From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving* (Jun 2026). Four axes (locality, lifetime, ownership, substrate); five archetypes (local-paged, disaggregated-pipeline, shared-store, memory-pool, hybrid-tier); seven named measurement gaps; the empty per-session-lifetime design point.
- `[C]` arXiv 2607.08057 — *Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization* (Jul 2026, ACL 2026 Findings). Temporal / spatial / structural framing.

**Code (verified `file:line`, `research/reference/CODE_MAP.md`)**
- `[C]` `memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:97`, `:658`; `core/block_pool.py:198`, `:647`, `:679`, `:702`, `:719`, docstring `:47-51`; `core/kv_cache_utils.py:184`, `:596`, `:691`; `core/kv_cache_manager.py:225`; `worker/block_table.py:81`; `attention/ops/triton_unified_attention.py:424`.
- `[C]` `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:217`, `:355`, `:565`, `:650`, `:676`, `:790`; `mem_cache/evict_policy.py:16`; `mem_cache/memory_pool.py:251`.
- `[C]` `memory/mooncake/mooncake-transfer-engine/include/transfer_engine.h:117`; `mooncake-store/include/master_service.h:382`; `include/replica_selection.h:122`; `include/allocation_strategy.h:402`; `include/storage_backend.h:354`; `src/master_service.cpp:5211`, `:6382`.
- `[C]` `memory/flashinfer/flashinfer/decode.py:710`, `:1239`, `:1481`; `flashinfer/page.py:326`, `:403`.
- `[C]` `architecture/llama-cpp-laguna/src/models/laguna.cpp:41`, `:184`; `src/llama-kv-cache-iswa.cpp:73`.
- `[C]` `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:365`, `:366`.

**Measured on our own instrument**
- `[M]` `notebook/uma-carveout-controls-fast-tier.md` (2026-07-26) — fast-tier boundary ≥62 GiB at ~200 GB/s with BIOS UMA FB Size = 96 GB; 30 GiB boundary with the 16 GiB default; single-buffer hang/fault at ≥32 GiB. Single run per arm.
- `[M]` `ASSUMPTIONS.md` rows `gpu-fast-tier-size`, `hardware-capacity-ceiling`, `large-tensor-fault-32gib`, `hipblaslt-config` (20.9 TFLOPS bf16 at 8192³), `gemm-throughput-below-reference`, `kv-per-token-laguna` (192 KiB/token upper bound), `laguna-heads-uniform` (refuted), `reference-model` (12 full + 36 sliding, GSSS, window 512), `single-device-only`.
- `[C]` ROCm issue #6034 (Mar 2026) — ~172 GB/s reported memory bandwidth for this silicon, cited in CLAUDE.md; our measured plateau of ~200 GB/s is consistent with it.

**Not cited because unverified.** Several vendor blog posts on 2026 KV-cache practice
surfaced during search with concrete-looking offload latency tables; none carried an
arXiv id or a reproducible harness, so their numbers are excluded rather than
repeated.
