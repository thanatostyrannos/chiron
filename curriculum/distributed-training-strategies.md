---
title: Distributed training strategies — FSDP, tensor, pipeline and expert parallel as four ways to spend bandwidth on memory
version: 1.0.0
date: 2026-07-26
track: D — Training systems
prereqs: the-training-loop, scaling-laws-and-flops-budget, moe-and-routing, attention-variants-and-kv-cost
also-helps: kv-cache-mechanics (for §4), memory-taxonomy-for-engineers (for §2.2)
difficulty: medium for the concepts, easy for the arithmetic, and unusually hard to stay honest about — none of it runs here
time: 3–4 h reading and working §2–§3 with a pen; 2–2.5 h for the exercises (A, the optional A′, B and C — all four were run before shipping; reference numbers are in the text)
---

# Distributed training strategies

**Read this paragraph before anything else.** Nothing in this module runs on the Z13. Not
"runs slowly", not "runs on one device with a warning" — does not run. `[M]` On the lab
wheel `torch 2.12.0a0+rocm7.13.0a20260313` (HIP 7.2.0, native Windows, gfx1151),
`torch.distributed.is_available()` returns **False**, the extension module
`torch._C._distributed_c10d` is **absent from the build**, and as a direct consequence
`torch.distributed.tensor` (DTensor), `torch.distributed.fsdp`, and
`torch.distributed.pipelining` all fail at *import* with
`ModuleNotFoundError: No module named 'torch._C._distributed_c10d'`. Reproduced in two
independent fresh processes, 2026-07-26; see Exercise A, which is how you check it yourself
in three minutes.

That is a stronger statement than the one this repo has been carrying. `ASSUMPTIONS.md →
single-device-only` and `research/memory/open-problems-ranked.md` both say collectives are
"incomplete" on gfx1151, tagged `[C]`. The local measurement says something more specific and
more consequential: the collectives are not *incomplete*, the whole `c10d` layer is **not
compiled in**, so even the parts of FSDP and DTensor that need no network at all — a
one-rank mesh, a CPU-only shard, a placement assertion — cannot be exercised in the lab venv.
You cannot write a unit test against `Shard(0)` here. You cannot import `DeviceMesh` to type
a function signature. That changes what "design-only" costs, and §4 is about paying that cost
deliberately instead of discovering it later.

So this module is **reading and design**. Its exercises are a capability probe, a
communication-volume calculator you write on paper and then in sixty lines of pure Python,
and a plan search you can do in your head if you like combinatorics. Every one of them
produces a number you can check. None of them touches the GPU, and there is no GPU variant to
fall back from, because the GPU cannot do the thing.

One precision, because "nothing runs" would otherwise be an overstatement. `[M]` The *other*
Python on this box — the non-lab CUDA build, `torch 2.11.0+cu128` — does have `torch.distributed`
with Gloo, and FSDP2 genuinely runs there **on CPU with two spawned ranks**: real shards, real
all-gathers, real reduce-scatters, countable. That is Exercise A′, it takes half an hour, and it
is the difference between believing §3.3's formula and watching it emit eight all-gathers. It is
also not the lab instrument and produces no evidence about our hardware. Both halves of that
sentence matter.

---

## 1. What this module settles

**One:** the four parallelism axes are one move played four ways — convert replicated storage
into communication — and they differ only in *what* they shard (the batch, the parameter
tensors, the layer stack, the expert set), which fixes *which collective* you pay and, more
importantly, *what the collective's volume is proportional to*: FSDP's traffic scales with
**parameters** and is independent of batch size; tensor and context parallel scale with
**tokens** and are independent of parameter count; pipeline parallel scales with tokens but
only across `p−1` seams; expert parallel scales with **tokens × top-k**, which is why an MoE
with `k=10` puts ten copies of every token's hidden state on the wire per layer. That single
table of proportionalities is the whole planning discipline.

**Two:** the received rules of thumb are dtype-conditional and the shipped defaults break
them. "FSDP costs 1.5× DDP's bandwidth" is true only when the gradient reduction runs in the
parameter dtype; torchtitan ships `mixed_precision_reduce: float32` against `bfloat16`
parameters (`configs.py:66`), and at those dtypes **FSDP and DDP move exactly the same number
of bytes per step** — 2.304 GiB per rank at our 353M-parameter reference arm on 8 ranks —
while FSDP's resident state is 674 MiB against DDP's 5.267 GiB. FSDP did not win on bandwidth.
It won on memory at bandwidth parity, and the standard framing hides that.

**Three:** for this lab the live consequence of Track D is not a parallel plan, it is a **config
surface and a checkpoint format**. Every parallel degree becomes a config field defaulting to
1 with the product invariant asserted; Mnemosyne must never import `torch.distributed` at
module scope, because on the lab wheel that import is fatal; and the checkpoint must be
sharded-by-parameter-name so that a single-device run can be resumed onto eight rented ranks
without a rewrite. Those three decisions cost nothing today and are irreversible later.

---

## 2. Theory in plain language

### 2.1 The single problem, and the four places you can cut

A training step needs, simultaneously resident on one device:

1. the **parameters** in the compute dtype,
2. their **gradients**,
3. the **optimizer state** (for AdamW: an fp32 master copy, a first moment, a second moment),
4. the **activations** saved by the forward pass for the backward pass to consume,
5. transient **workspace** — the logits tensor, attention scores, communication buffers.

Items 1–3 are proportional to parameter count and independent of batch size. Item 4 is
proportional to tokens-in-flight and independent of parameter count *except* through depth.
Two different scaling laws in one memory budget, which is why there are two different families
of parallelism.

When the sum exceeds one device, you have exactly four cuts available, and they are the four
axes of the model's computation graph:

| Cut | Axis | Name | What each rank ends up holding |
|---|---|---|---|
| Cut the **batch** | data | DP / DDP | a whole model, a slice of the batch |
| Cut the **parameter tensors** | width | FSDP (ZeRO) or TP | a slice of every weight matrix |
| Cut the **layer stack** | depth | PP | a contiguous run of whole layers |
| Cut the **expert set** | conditional width | EP | a subset of the experts, all of the other layers |

There is a fifth cut — the **sequence** — which is context parallelism (CP). It is not in this
module's title but it appears throughout, because it is the one that touches Mnemosyne
directly: CP shards the KV cache along position.

Every cut buys memory and costs a collective. That trade is the entire subject. The design
question is never "should I use FSDP" — it is "which cut has the smallest communication
volume *for my particular shape*, and does its collective fit on the link I have."

### 2.2 The bridge you already own, and three places it breaks

> **Systems bridge.** This is sharding and replication. `N` nodes, a keyspace, a placement
> function, a rebalance cost, a consistency model. You have designed this system. FSDP is
> range-sharding a keyspace of parameters; TP is striping a single record across devices; PP
> is a store-and-forward processing pipeline; EP is a router in front of a sharded keyspace.

The bridge carries a long way. Then it breaks in three specific places, and each break is a
thing the ML literature assumes you know and never states.

**Break one: a shard here is not independently useful.** In every sharded store you have
built, a shard can *serve*. Give it a key in its range and it answers. A shard of a weight
matrix answers nothing. You cannot compute `x·W` from a row-slice of `W` without the matching
slice of `x`, and after the matmul you hold a partial sum that is not the answer. So the
shards must be reassembled **before every use, on every step, forever**. The all-gather is not
a cache miss to be minimized toward zero — it is the steady state. FSDP is not a partitioning
of work. It is a partitioning of *idle storage* for a computation that stays global.

The practical corollary, which the code makes concrete: FSDP's memory saving is real only
between uses. `fully_shard` is applied per transformer block
(`torchtitan/distributed/fsdp.py:180`), so at any instant one block's parameters are fully
materialised on every rank while the rest are sharded. Peak memory is therefore
`sharded_total + one_block_gathered × (1 + prefetch_depth)`, not `sharded_total`. Read
`reshard_after_forward` (`fsdp.py:147`) as the knob that chooses whether to free the gathered
copy after forward and pay a second all-gather in backward, or keep it and pay memory —
ZeRO-3 versus ZeRO-2, and the FSDP2 documentation states that mapping explicitly
(`docs/fsdp.md:60`).

**Break two: replication makes the job strictly *less* available.** You replicate for
availability. Data parallelism replicates for *statistical* reasons only — averaging gradients
over a larger sample. There is no failover. A synchronous collective completes when the last
rank arrives; a dead rank means the collective never completes, and the job dies. With `N`
ranks each with independent failure probability `f` per unit time, job survival is `(1−f)^N`.
**Availability decreases monotonically in the number of replicas.** That is the exact inverse
of every replication system you have run, and it is why frontier training reports read like
site-reliability postmortems: `[C]` MegaScale (2402.15627) is largely a paper about detecting
and evicting stragglers, and `[C]` the Llama 3 report (2407.21783) documents failure rates
that make checkpoint interval a first-order design parameter, not an operational detail.

**Break three: there is no consistency model, because there is no concurrency.** No eventual
consistency, no read-your-writes, no quorum, no vector clocks. Every collective is a global
barrier. Between barriers the ranks execute *the same program on different data*, in lockstep,
and any divergence in control flow deadlocks rather than degrades. The system is a SIMD
machine whose lanes happen to be connected by a network. "Distributed" describes the topology,
not the semantics — and the single most common bug in this domain is a rank taking a different
branch (a different number of microbatches, a different early-exit, a `if rank == 0` around a
collective) and hanging the job with no error, which is the same failure signature as
`ASSUMPTIONS.md → large-tensor-fault-32gib`: a stall at zero CPU, not a crash.

### 2.3 What each axis replaced

Knowing what a technique *displaced* tells you what it is actually for.

**DDP replaced the parameter server.** Early distributed SGD used a central server holding the
weights; workers pushed gradients and pulled parameters. That is a star topology with a
bandwidth hotspot at the centre and a consistency question (synchronous? stale? Hogwild?).
`[C]` PyTorch DDP (2006.15704) replaced it with a ring all-reduce: no centre, bandwidth per
rank independent of `N`, and strictly synchronous semantics. The field decided, around
2017–2018, that asynchronous updates were not worth the accuracy risk, and that decision is
now being reopened — see §8.

**FSDP replaced "the model must fit."** `[C]` ZeRO (1910.02054) observed that DDP replicates
three things that are only *read* once per step — optimizer state, gradients, parameters — and
shards them in three stages. `[C]` PyTorch FSDP (2304.11277) is the in-framework implementation;
FSDP2 (the current one) replaced FSDP1's `FlatParameter` — a concatenated blob per
communication bucket — with per-parameter `DTensor`s sharded on dimension 0, which is what
makes per-parameter mixed precision, communication-free sharded state dicts, and meta-device
initialisation possible (`docs/fsdp.md:9`). If you have wondered why FSDP checkpoints used to
be a nightmare, that is the answer: the old format's unit of storage was a bucket, not a
tensor.

**TP replaced nothing; it was the original.** `[C]` Megatron-LM (1909.08053) split the weight
matrices themselves across devices, exploiting the fact that an MLP `Y = GeLU(XA)B` can be
column-split on `A` and row-split on `B` so that **no synchronisation is needed between the two
GEMMs**: the first produces a sharded output that is exactly the input the second wants, and one
all-reduce at the end of the block suffices instead of one after each matrix. The same pairing
applies to attention, with `Wq/Wk/Wv` column-split by head and `Wo` row-split. `[C]` The
follow-up (2205.05198) added sequence parallelism, which shards the
non-matmul regions (layer norm, dropout) along sequence and replaces each all-reduce with a
reduce-scatter plus an all-gather of equal total volume — free memory, same bytes.

**PP replaced naive model-splitting.** Cutting a model at layer 12 and putting the halves on
two devices gives you 50% utilisation: one device idles while the other works. `[C]` GPipe
(1811.06965) fixed the utilisation by splitting the batch into microbatches and filling the
pipe, at the cost of holding every microbatch's activations until its backward pass. `[C]`
PipeDream (1806.03377) introduced 1F1B — one forward, one backward, interleaved — which has the
same bubble ratio but caps in-flight activations at the number of stages rather than the number
of microbatches. `[C]` Zero Bubble (2401.10241) then observed that the backward pass has two
halves — the input gradient, which the previous stage needs immediately, and the weight
gradient, which nothing needs until the optimizer step — and deferred the second into the
bubble.

**EP replaced "every device holds every expert."** `[C]` GShard (2006.16668) and `[C]` Switch
Transformer (2101.03961) put different experts on different devices and shuffled tokens to
their assigned experts with an all-to-all. This is the only axis whose collective's *shape* is
decided by the model at runtime, which is the subject of §3.6.

---

## 3. The math that actually matters

### 3.1 Symbols, every one in words

| Symbol | In words | Units |
|---|---|---|
| `P` | number of trainable parameters — in the whole model, or in one pipeline stage where noted | count |
| `b_p` | bytes per element of the dtype the parameters are *gathered and computed* in; 2 for bf16 | B |
| `b_r` | bytes per element of the dtype the gradient *reduction* runs in; 4 for fp32 | B |
| `N` | number of ranks participating in one particular collective (not the world size) | count |
| `L` | number of transformer layers | count |
| `d` | model width — the residual-stream / hidden size | count |
| `T` | tokens fed to one rank's forward pass per optimizer step | count |
| `m` | microbatches per optimizer step | count |
| `p` | pipeline stages | count |
| `v` | virtual stages held per rank (interleaving factor); 1 for non-interleaved | count |
| `E` | number of experts in an MoE layer | count |
| `k` | experts activated per token (top-k) | count |
| `ep` | expert-parallel degree — number of ranks the experts are spread over | count |
| `α` | fixed cost of one hop of a collective: link latency plus per-message software cost | s |
| `β` | seconds per byte on the link; `β = 1 / bandwidth` | s/B |
| `M` | size of the message a collective operates on | B |

Throughout, "bytes per rank per step" means bytes crossing one rank's own link, in one
direction, summed over the whole optimizer step. That is the quantity that divides by link
bandwidth to give a *lower bound* on time. It is not a prediction — see §3.8.

### 3.2 The cost of a collective: the α–β model

For ring algorithms, which is what NCCL and RCCL use for large messages, with `N` ranks and a
message of `M` bytes:

```
all-reduce      time = 2(N−1)·α  +  2·((N−1)/N)·M·β
all-gather      time =  (N−1)·α  +      ((N−1)/N)·M·β
reduce-scatter  time =  (N−1)·α  +      ((N−1)/N)·M·β
all-to-all      time ≈  (N−1)·α  +      ((N−1)/N)·M·β
```

- `2(N−1)·α` — a ring all-reduce is a reduce-scatter followed by an all-gather, each of which
  makes `N−1` hops. This term is **latency**, and it grows with `N`.
- `2·((N−1)/N)·M·β` — the bytes. As `N` grows this approaches `2M`, i.e. **the byte cost of a
  ring all-reduce is asymptotically independent of the number of ranks.** That is the property
  that makes data parallelism scale at all.
- For all-gather and reduce-scatter, `M` is the size of the *full* (gathered) tensor.
- For all-to-all, `M` is the size of one rank's *outgoing* buffer, and the `(N−1)/N` is the
  fraction not destined for itself.

Two consequences worth internalising immediately. First, **big messages are bandwidth-bound
and small messages are latency-bound**, which is why gradient reduction is bucketed —
Megatron's `_ParamAndGradBucket` (`param_and_grad_buffer.py:92`) coalesces parameters into
buckets of roughly `bucket_size` elements (`param_and_grad_buffer.py:979`) so that one
collective carries many tensors. This is write-combining, and it is the same reason your
storage layer batches. Second, for small `N` the `(N−1)/N` factor matters: at `N=2` you move
half as many bytes as the asymptote, at `N=8` you move 87.5%.

> **Systems bridge.** `α` and `β` are exactly service time and transfer time in a queueing
> model, and the bucketing decision is exactly the "coalesce small writes" decision.
>
> **Where it breaks.** In your storage layer, a batched write that misses its deadline is
> late. Here, the collective is a *barrier*: every rank blocks until the slowest finishes, so
> the cost is not the mean, it is the maximum over ranks, every single step, with no
> smoothing. A 1%-of-the-time straggler on one rank out of 1024 is not a 1% tail — it is a
> near-certainty that *some* rank is slow on any given step.

### 3.3 FSDP: the memory ledger, and the bandwidth surprise

**The ledger.** AdamW in mixed precision, per parameter, in the standard recipe:

```
bf16 parameter   2 B
bf16 gradient    2 B
fp32 master      4 B
Adam m           4 B
Adam v           4 B
                ────
                16 B / parameter
```

ZeRO shards these in stages. With `N` data-parallel ranks, per-rank resident bytes per
parameter:

```
DDP    (nothing sharded)          16
ZeRO-1 (optimizer state: 12 B)     4 + 12/N
ZeRO-2 (+ gradients: 14 B)         2 + 14/N
ZeRO-3 (+ parameters: 16 B)            16/N
```

At our reference arm — a dense Proteus-scale decoder, `L=24`, `d=1024`, vocabulary 50,257,
FFN width `4d`, tied embeddings, so `P = 353,453,056` total (301,989,888 non-embedding,
12,582,912 per block) — and `N=8`:

| Scheme | B/param | Resident per rank |
|---|---|---|
| DDP | 16.00 | **5.267 GiB** |
| ZeRO-1 | 5.50 | 1.810 GiB |
| ZeRO-2 | 3.75 | 1.234 GiB |
| ZeRO-3 / FSDP | 2.00 | **674.158 MiB** |

plus the transient all-gather buffer: one block in bf16 is `2 × 12,582,912 = 24.000 MiB`, or
48.000 MiB when the next block is prefetched (which torchtitan does explicitly under EP —
`fsdp.py:299` onward — and implicitly otherwise).

**Now hold that against our hardware.** `[M]` The Z13's fast memory tier is **≥62 GiB at
~200 GB/s** (`ASSUMPTIONS.md → gpu-fast-tier-size`). The *fully replicated* 16 B/param state
for this model is 5.267 GiB — **8.4% of one device's fast tier**. FSDP's entire reason to exist
does not apply to us at this scale. We are not memory-poor; at 300M parameters we have roughly
eleven times the memory we need for the replicated ledger, which is the whole argument in
`research/notes/pretraining-recipes.md` §5 restated at the collective level. Remember this
when reading FSDP tutorials: they are written for someone with 80 GB and a 70B model. We have
128 GB and a 0.3B model. The advice inverts.

**The bandwidth surprise.** Per optimizer step, per rank, FSDP moves:

```
forward all-gather of parameters      P · b_p · (N−1)/N
backward all-gather of parameters     P · b_p · (N−1)/N      (only if reshard_after_forward)
backward reduce-scatter of gradients  P · b_r · (N−1)/N
```

so

```
FSDP (ZeRO-3)  = (2·b_p + b_r) · P · (N−1)/N
FSDP (ZeRO-2)  = (  b_p + b_r) · P · (N−1)/N
DDP            = (      2·b_r) · P · (N−1)/N        (ring all-reduce)
```

That "two all-gathers and one reduce-scatter per FSDP unit" is not an inference from the
papers; you can count the ops. `[M]` A four-block model sharded on a two-rank CPU mesh emits
exactly **8 `_allgather_base_` and 4 `_reduce_scatter_base_`** calls in one forward-plus-backward
— see Exercise A′, which is the only executable thing in this module and runs on a non-lab
interpreter.

Everyone quotes the ratio at `b_p = b_r`, where FSDP-3 : DDP = `3 : 2` = **1.5×**. But
torchtitan's shipped configuration is `mixed_precision_param = bfloat16` and
`mixed_precision_reduce = float32` — and the type annotation on the latter is
`Literal["float32"]`, i.e. it is not even configurable (`configs.py:66`). Substituting
`b_p = 2, b_r = 4`:

```
FSDP-3 = (2·2 + 4)·P·(N−1)/N = 8·P·(N−1)/N
DDP    = (    2·4)·P·(N−1)/N = 8·P·(N−1)/N
```

**Identical.** At our reference arm on 8 ranks: DDP 2.304 GiB/rank/step, FSDP-3
2.304 GiB/rank/step, ratio 1.000; FSDP-2 (`reshard_after_forward=False`) 1.728 GiB, ratio
0.750 — *cheaper* than DDP. Switch the reduction to bf16 and you recover the textbook numbers:
DDP 1.152 GiB, FSDP-3 1.728 GiB, ratio 1.500.

So: **FSDP is not a bandwidth cost you pay for memory. At the shipped defaults it is a memory
saving at bandwidth parity.** The "1.5×" rule of thumb is a statement about dtypes, not about
FSDP, and if you carry it into a plan decision you will over-weight DDP by 50%.

Two details in the code that a reader with your background will want and will not find in the
papers:

- `disable_fsdp_gradient_division` (`fsdp.py:41`) sets `gradient_divide_factor=1.0`. FSDP by
  default folds a `1/N` into the reduce-scatter so the result is a mean. torchtitan turns that
  off because it divides the loss by the *global token count* before backward instead — the
  same choice OLMo-core makes, and for the same reason: with ragged microbatches, a mean of
  means is not the mean. `research/notes/pretraining-recipes.md` §2 makes this point at the
  accumulation level; this is the same correctness argument one layer down, at the collective.
- `get_fsdp_reshard_after_forward_policy` returns `not pp_enabled` by default
  (`fsdp.py:74`). Under pipeline parallelism, each stage runs forward once per *microbatch*, so
  resharding after forward would issue an all-gather per microbatch instead of per step. This
  is the first concrete instance of the rule that the axes are not independent: turning on PP
  silently changes FSDP's memory/bandwidth trade.

### 3.4 Tensor parallel: the volume that scales with tokens

Megatron's transformer block, TP degree `N`:

- **Attention.** `Wq`, `Wk`, `Wv` are column-sharded (each rank owns a subset of heads);
  `Wo` is row-sharded. Read `ColumnParallelLinear` (`layers.py:869`) — its docstring states
  the split precisely: `Y = XA + b` with `A = [A_1, ..., A_p]` along the second dimension —
  and `RowParallelLinear` (`layers.py:1249`), where `A` is split along the first dimension and
  the input `X` along its second.
- **MLP.** `W1`/`W3` column-sharded, `W2` row-sharded.

The pairing is the trick: a column-sharded layer needs a *replicated* input and produces a
*sharded* output; a row-sharded layer consumes a sharded input and produces a **partial sum**
that must be all-reduced. So one all-reduce per sub-block in forward, and one in backward on
the input gradient at the column-parallel entry. torchtitan expresses exactly this
declaratively: `colwise_config()` sets weight placement `Shard(0)` and output `Shard(-1)`
(`decoder_sharding.py:65`); `rowwise_config()` sets weight `Shard(1)` and — this is the line to
stare at — `out_src_shardings=dense_activation_placement(tp=spmd.P)`
(`decoder_sharding.py:91`), where `P` is *Partial*: the output is not a tensor, it is a
promise of a sum.

Per layer, per step, four all-reduces of the activation tensor `[tokens, d]`:

```
TP bytes / rank / step  =  L · 4 · 2 · (T · d · b_p) · (N−1)/N
```

- `L` — layers, because every layer pays.
- `4` — two all-reduces forward (attention output, MLP output), two backward.
- `2` — the ring all-reduce factor from §3.2.
- `T · d · b_p` — the activation tensor: tokens × width × bytes. **Note what is absent:
  parameter count.**

At the reference arm with 65,536 tokens per step and `N = 8`:

```
TP(8)   = 24 · 4 · 2 · (65,536 · 1024 · 2) · 7/8  =  21.000 GiB / rank / step
FSDP(8) =                                            2.304 GiB / rank / step
```

TP moves **9.1×** what FSDP moves. Set the two expressions equal and solve for `T`: the
crossover is at **7,191 tokens per step**. Above that — which is every real configuration —
tensor parallelism is the more expensive axis by bandwidth, and it is expensive *per layer, on
the critical path*, with no opportunity to overlap the second all-reduce of a block with
anything, because the next layer depends on it.

That is the whole reason TP is confined inside a node. At 900 GB/s (NVLink) 21 GiB is 25.0 ms;
at 50 GB/s (a fast Ethernet fabric) it is 451 ms. For scale, a step of this model on 8 ranks
at 350 TFLOP/s sustained is about 49.6 ms of compute (`6·N·D` = `6 × 353,453,056 ×
8,192` = 1.74e13 FLOP per rank). So TP-8 over NVLink costs ~50% overhead and TP-8 over
Ethernet costs 9× the step. **TP does not cross the node boundary.** That is not a convention,
it is this arithmetic.

Sequence parallelism does not change the byte count. It replaces each all-reduce with a
reduce-scatter plus an all-gather (`decoder_sharding.py:53` and the `output_sp` branch of
`rowwise_config`), which sums to the same `2M(N−1)/N`, and in exchange the layer-norm and
dropout regions hold `1/N` of the activations. Free memory, identical bandwidth. `[C]`
2205.05198.

> **Systems bridge.** TP is RAID-0 striping *within a record*. Every read touches every disk;
> stripe width is a tuning parameter; you would never stripe across a WAN.
>
> **Where it breaks, and this one matters for us.** A striped read returns identical bytes at
> any stripe width. A TP forward does not. The row-parallel output is a **floating-point sum
> over `N` partial products**, and changing `N` changes the summation tree. Reassociation of
> floating-point addition is not a no-op. `[M]` On this machine we have already measured that
> the reduction path is worth **2.8× in relative error** on a length-1,048,576 bf16 weighted
> sum — 2.01e-3 with hipBLASLt configured versus 5.60e-3 without
> (`ASSUMPTIONS.md → hipblaslt-config`, 3 seeds, fresh processes). TP degree is the same class
> of perturbation applied to a much shorter reduction. So **the TP degree is a numerics axis,
> not only a performance axis**, and a run at TP=1 and a run at TP=8 are not the same
> experiment. torchtitan's own contributor rules say to "verify numerics match before and
> after across multiple parallelism configs" — an instruction, which means it is not a
> guarantee.

### 3.5 Pipeline parallel: the cheapest bytes and the most expensive idleness

**Bytes.** Only activations cross a stage boundary, and only at the `p−1` seams. Per boundary
per step, forward activations plus backward gradients:

```
PP bytes / boundary / step  =  2 · T · d · b_p
```

and note what is *not* in it: `m`, the microbatch count. Splitting the batch into more
microbatches sends the same total bytes in more, smaller messages — it changes the `α` term,
not the `β` term. At the reference arm, 65,536 tokens, `p=4`: **768.000 MiB** total across the
three seams, against TP-8's 21.000 GiB per rank. The ratio is **28×**. Pipeline parallel is,
by a wide margin, the cheapest axis in bytes, which is precisely why it is the axis you run
across the slowest link.

**Idleness.** What PP costs instead is a bubble. With `p` stages and `m` microbatches, the
pipeline must fill and drain once per optimizer step, because synchronous semantics require
every microbatch's gradient before the step:

```
bubble fraction = (p − 1) / (v·m + p − 1)
```

- `p − 1` — the number of stage-times spent filling, and again draining, expressed as a
  fraction of the whole.
- `v` — virtual stages per rank. Interleaving cuts each rank's work into `v` non-contiguous
  chunks so the fill overlaps; Megatron computes the warm-up count as
  `(p − rank − 1) · 2 + (chunks − 1) · group_size` (`schedules.py:929`) against the plain
  1F1B `p − rank − 1` (`schedules.py:922`). Read those two lines side by side; they are the
  entire difference between the schedules.
- `m` — microbatches. More microbatches shrink the bubble and are free in bytes.

| `p` | `m` | bubble, `v=1` | bubble, `v=2` |
|---|---|---|---|
| 4 | 8 | 0.273 | 0.158 |
| 4 | 32 | 0.086 | 0.045 |
| 8 | 8 | 0.467 | 0.304 |
| 8 | 64 | 0.099 | 0.052 |

torchtitan warns when `m < p·stages_per_rank` (`pipeline_parallel.py:263`) — below that you are
paying more than half your pipeline in bubble.

> **Systems bridge.** An assembly line. Stages, work-in-progress, throughput set by the
> slowest station, fill and drain at shift change.
>
> **Where it breaks, three ways.** (1) **The line runs backwards too.** Every item traverses
> forward and then in reverse, and the reverse traversal needs the forward traversal's
> intermediate state still resident. So this is an assembly line where every station must
> *retain* every item it has touched until the item comes back. That retention is the real cost
> of PP, and it is why 1F1B exists: GPipe holds `m` microbatches' activations at stage 0, 1F1B
> holds at most `p`. (2) **You cannot cut the line where the work is.** You cut where a module
> boundary is, and the boundary must preserve fully-qualified parameter names or distributed
> checkpointing breaks — torchtitan switched `layers` from a `ModuleList` to a `ModuleDict`
> specifically so that deleting `layers.0` leaves `layers.1` named `layers.1`
> (`docs/composability.md:14`). The splitter is `copy.deepcopy` the whole model and delete what
> this stage does not own (`pipeline_parallel.py:446`), which is blunt and readable and tells
> you the boundary is a *naming* problem, not a graph problem. (3) **The drain happens every
> step.** A fill/drain you amortise over a long run would be free. This one recurs per
> optimizer step because the semantics are synchronous. It is the cost of committing every
> transaction with a full pipeline flush.

One more PP fact with no analogue in your world: **initialisation.** Each stage holds only its
own layers, and you want bitwise-identical initialisation to a single-device run for debugging.
torchtitan's answer is brutal and honest — initialise the whole model on a CPU box, save a
"seed checkpoint", and let distributed checkpointing load the subset of names each stage owns
(`docs/composability.md:19`). Random-number-generator order is a global side effect and there
is no clean way to shard it.

### 3.6 Expert parallel: the collective whose shape the model chooses

Each MoE layer, `ep` ranks, top-`k` routing:

```
EP bytes / rank / step  =  L_moe · 4 · (T · k · d · b_p) · (ep−1)/ep
```

- `L_moe` — the number of MoE layers.
- `4` — dispatch and combine in forward, and both again in backward.
- `T · k · d · b_p` — each of this rank's `T` tokens sends its `d`-wide hidden state to `k`
  experts. **The `k` is the amplification and it is the whole story.**
- `(ep−1)/ep` — with balanced routing, the fraction of tokens whose expert lives elsewhere.

Put Laguna S 2.1's shipped numbers in: `d = 3072`, `k = 10`, `E = 256`, bf16 (`[M]`
`ASSUMPTIONS.md → reference-model`; `config.json` at revision `b0a9fd7c850e`,
`hidden_size: 3072`, `num_experts: 256`, `num_experts_per_tok: 10`).

`L_moe` is **47, not 48** — and this is a small worked example of why the house rule is to read
the artifact rather than quote the architecture. The config carries `mlp_only_layers: [0]` and
an explicit `mlp_layer_types` array whose first entry is `"dense"` and whose remaining 47 are
`"sparse"`; `models/laguna-s/modeling_laguna.py:480` is the branch that consumes both. Layer 0
is a plain dense MLP. My own first pass through this arithmetic used 48 and was 2.1% high.
There is also an always-on `shared_expert` (`shared_expert_intermediate_size: 1024`,
`models/laguna-s/modeling_laguna.py:246`) whose work is *local to every rank* and therefore
contributes nothing to the shuffle — a fraction of the FFN that is structurally EP-free.

Per token, per MoE layer, on the wire:

```
10 · 3072 · 2 B = 60.000 KiB      versus the token's own hidden state, 6.000 KiB
```

**A 10× amplification, per layer, on 47 of 48 layers.** At 8,192 tokens per rank per step:
`ep=8` moves **77.109 GiB per rank per step**; `ep=16` moves 82.617 GiB; `ep=64` moves
86.748 GiB. Note the shape of that: an 8× increase in device count adds **12.5%** to the
volume, because the `(ep−1)/ep` factor is already 0.875 at `ep=8` and saturates at 1 — a
counterintuitive result the formula hands you for free. A single dispatch all-to-all at
`ep=64` is 472.500 MiB.

Compare: the whole dense-arm FSDP traffic was 2.304 GiB. **Expert parallelism is not a
comparable axis; it is one to two orders of magnitude larger, and it is why the MoE systems
literature is almost entirely about all-to-all.** Recent measurements put all-to-all at 34–40%
of MoE training step time `[C]` (MegaScale-MoE, 2505.11432; DisagMoE, 2605.11005).

> **Systems bridge.** A router in front of a sharded keyspace. Consistent hashing, a shuffle
> phase, and a load-balancing problem — this is MapReduce's shuffle, and you have tuned one.
>
> **Where it breaks, three ways, and all three are unusual.**
>
> **(1) The hash function is learned and it drifts.** The router is a trained linear layer
> (`architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:183`, sigmoid
> scoring). Your placement function changes during the run.
> And you cannot rebalance by rehashing, because moving a token to a different expert changes
> what the model computes. The only levers are to change the *loss* (an auxiliary
> load-balancing term) or to nudge selection with a learned per-expert bias that does not
> affect the combination weights — which is exactly what Laguna does
> (`architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:185`,
> `e_score_correction_bias`, with `router_aux_loss_coef = 0.0`).
> Load balancing has been moved out of the systems layer and into the objective function.
>
> **(2) The collective's descriptor is data-dependent.** Every other collective in the plan has
> a shape known before the step begins. The EP all-to-all does not: how many tokens go to each
> rank is the router's output. torchtitan must first exchange the *counts* with a small
> all-to-all (`token_dispatcher.py:273`), then copy the resulting split sizes to the host —
> and the code says so plainly: `# NOTE: this would incur a device-to-host sync`
> (`token_dispatcher.py:313`) — before it can issue the variable-size data all-to-all
> (`token_dispatcher.py:340`) and its mirror on the combine side (`token_dispatcher.py:368`).
> A mandatory host synchronisation in the middle of the forward pass, once per MoE layer. This
> is also why torchtitan switches FSDP to *explicit* prefetching when EP is on: the D2H syncs
> interfere with FSDP's implicit prefetch heuristics (`fsdp.py:297`).
>
> **(3) The straggler is structural, and the industry's fix is a correctness compromise.** An
> all-to-all completes when the slowest pair completes. The number of tokens routed to the
> busiest expert is a random variable with a heavy tail. The standard mitigation is a *capacity
> factor* — a hard cap per expert, above which tokens are **dropped**, meaning they skip the
> MoE layer entirely and pass through the residual. No load balancer you have ever operated is
> permitted to discard requests to hit a latency target. This one is, and it is the default in
> several production stacks.

There is a fourth hazard that only shows up when you compose EP with activation checkpointing,
and it is the best single example in this module of why the axes are not independent. Selective
activation checkpointing recomputes the forward pass during backward. If it recomputes the
router, `torch.topk` can return a *different* expert assignment (ties, non-determinism), so the
backward pass would attribute gradients to experts that never ran. torchtitan therefore puts
`torch.ops.aten.topk.default` on the must-save list with the comment "topk can be
non-deterministic; save to keep MoE expert assignments stable between forward and recompute"
(`activation_checkpoint.py:53`), and separately saves the outputs of
`reduce_scatter_tensor` and `all_to_all_single` so that recompute does not re-issue the
collectives (`activation_checkpoint.py:62`, `:63`). A memory optimisation, a communication
optimisation, and a routing mechanism, all three of which are individually correct and jointly
wrong unless someone writes down that list.

### 3.7 Composition: the product invariant, and what it costs to add an axis

torchtitan's `ParallelDims._validate` is the whole composition contract in one assertion
(`parallel_dims.py:181`):

```
dp_replicate · dp_shard · cp · tp · pp == world_size
```

Five axes whose degrees must multiply to the device count. Note that `ep` is **not** in the
product. Expert parallelism is carved *out of* the other axes, not added to them
(`parallel_dims.py:274`):

```
efsdp = (dp_shard · cp · tp) / ep
```

so turning on EP shrinks the FSDP group that shards the *expert* weights, while the dense
weights keep the full group. Two different data-parallel meshes over the same devices, which
is why `apply_fsdp_to_decoder` needs a per-parameter `shard_placement_fn` returning a different
`mesh_info` for expert parameters than for everything else (`fsdp.py:252`).

The plan also constrains the *data*, not just the model (`parallel_dims.py:701`):

```
seq_len must be divisible by  tp · (cp · 2)
```

`tp` because sequence parallelism shards the sequence across TP ranks; `cp · 2` because
context parallelism's default load balancer pairs a head chunk with a tail chunk, so it needs
an even number of chunks per rank. That factor of two is the causal-mask workload imbalance
made into a divisibility rule.

**How many plans are there?** If the world size is `2^w`, the number of ordered 5-tuples of
powers of two multiplying to `2^w` is the number of ways to write `w` as an ordered sum of five
non-negative integers:

```
number of plans = C(w + 4, 4)
```

At `world = 8` (`w=3`): **35**. At 64: 210. At 1024: 1001. And that is only powers of two, only
five axes, and ignoring schedule choice, `reshard_after_forward`, activation-checkpointing
policy, and microbatch count. The search space is why §8 lists automatic planning as unsolved.

**The ordering heuristic**, which every production stack follows and none of them derives:
place the axis with the highest byte volume on the fastest link.

```
innermost (fastest link)   TP        L · 4 · 2 · T·d·b · (N−1)/N     — per layer, critical path
                           EP        L_moe · 4 · T·k·d·b · (ep−1)/ep — per MoE layer, variable size
                           CP        per layer, KV-sized
                           FSDP      (2b_p + b_r) · P · (N−1)/N      — per step, overlappable
outermost (slowest link)   DP repl   2·b_r · P · (N−1)/N             — per step, once
                           PP        2 · T·d·b per seam              — per step, tiny
```

HSDP is this heuristic applied to a two-level network: shard within a node where bandwidth is
high, replicate across nodes where it is not. OLMo-core's default states it in one line —
one replica per node, shard degree equal to GPUs per node
(`olmo_core/distributed/parallel/data_parallel.py:49`).

### 3.8 What the byte model does not know

Everything in §3.3–§3.7 counts bytes. Four things it deliberately ignores, and you must hold
all four or the model will lie to you:

1. **Overlap.** FSDP's all-gather for block `i+1` can run under block `i`'s compute. TP's
   all-reduce mostly cannot, because the next operation depends on it. Two plans with identical
   byte counts can differ by 2× in time.
2. **Latency.** The `α` term. At small `N` and small messages, latency dominates and the byte
   model says nothing.
3. **Memory feasibility.** The byte model happily recommends `pp=8, dp=1` and never notices
   that stage 0 has to hold eight microbatches of activations.
4. **The bubble.** PP's bytes are tiny; its idleness is not. Rank purely by bytes and PP always
   wins, which is exactly why nobody uses pure PP.

Exercise C makes this concrete: the byte-optimal plan at `world=8` is `pp=4`, and it is
byte-optimal for a reason the byte model cannot see through.

---

## 4. Why this matters for Proteus and Mnemosyne

None of this runs here. So the question is not "which plan" but "what must be true of our code
today so that a plan is a config change later, and what must be true so that Mnemosyne is a
contribution rather than a single-device artifact." Six answers, in decreasing order of how
expensive they are to retrofit.

### 4.1 The KV cache has three incompatible partitionings, and Mnemosyne must name them

This is the point where Track D collides with Track C, and it is the most important paragraph
in this module for the research programme.

Under a 4D plan the KV cache is sharded three different ways at once:

| Axis | What the KV cache is split by | Per-rank shape |
|---|---|---|
| TP | **heads** | `n_kv / tp` heads, all positions, all layers on this stage |
| CP | **sequence position** | all heads, `T / cp` positions |
| PP | **layers** | all heads, all positions, `L / pp` layers |

An addressing scheme that says "block `b` holds positions `[i, i+16)` of layer `ℓ`" is
single-device by construction. Under TP it must also say *which heads*; under CP it must say
which rank owns which positions, and the CP load balancer means those positions are **not
contiguous** — `_HeadTailLoadBalancer` (`context_parallel.py:222`) deliberately gives each
rank one chunk from the front and one from the back, because under causal masking the last
chunk attends to everything and the first attends to almost nothing. Range-partitioning a
monotone workload requires the balancer to be part of the partitioner.

Two consequences for Mnemosyne's interface:

- **The cache's coordinate space must be a declared layout, not an implicit convention.** A
  small owned dataclass — `(layer_range, head_range, position_set, dtype, page_size)` — that
  degenerates to the whole model on one device. Eviction policies then operate on *local*
  coordinates and the layout tells them what the local view means. Without this, an eviction
  policy validated at TP=1 silently means something different at TP=8, because "the 100 tokens
  with lowest attention mass" is now "…as seen by this rank's subset of heads."
- **GQA and TP collide.** Laguna has `num_key_value_heads = 8`, uniform across all 48 layers
  `[M]` (`ASSUMPTIONS.md → laguna-heads-uniform`). At `tp > 8` there are not enough KV heads to
  shard, and the standard fix is to *replicate* KV heads across TP ranks — which means the KV
  cache stops shrinking with TP degree and starts costing `tp/n_kv` times more in aggregate.
  Any Mnemosyne cost model keyed on `n_kv` alone is wrong above `tp = n_kv`.

There is a research gap here that is directly adjacent to our stated contribution: **no
published work characterises how KV-eviction policies behave under head-sharding or under
non-contiguous sequence-sharding.** `research/memory/open-problems-ranked.md` parks distributed
KV tiering because T ≤ 2 on our hardware. This is the *design-time* half of the same question,
and it costs nothing to get right now.

### 4.2 The import that kills the package

The boundary rule in `CLAUDE.md` is `mnemosyne → torch`, and `torch.distributed` is a torch
submodule, so importing it does not violate the dependency graph. It nevertheless **must not
appear at module scope in Mnemosyne**, because `[M]` on the lab wheel that import raises
`ModuleNotFoundError` — a module-level `from torch.distributed.tensor import DTensor` makes the
whole package unimportable on the machine we develop on. The rule to write down now:

- Mnemosyne owns a plain-dataclass layout descriptor and never imports `torch.distributed`.
- Any DTensor / DeviceMesh adapter lives in Proteus or Themis, behind a function-scope import,
  guarded by `torch.distributed.is_available()`.
- The clean-venv acceptance test at the `mnemosyne-core` milestone should assert the absence:
  a test that fails if `torch.distributed` appears in Mnemosyne's import graph. That is the
  same shape of guard as `tests/test_package_boundaries.py`, which has already been proved
  red-then-green `[M]` (`ASSUMPTIONS.md → mnemosyne-separable`).

### 4.3 The config surface is the experimental surface, even for experiments we cannot run

Add now, default 1, validate always:

```yaml
parallelism:
  data_parallel_replicate_degree: 1
  data_parallel_shard_degree: 1
  context_parallel_degree: 1
  tensor_parallel_degree: 1
  pipeline_parallel_degree: 1
  expert_parallel_degree: 1
```

with two validators, both TDD-able today on one device and both cheap:

1. `dp_replicate · dp_shard · cp · tp · pp == world_size` (`parallel_dims.py:181`).
2. `seq_len % (tp · cp · 2) == 0` (`parallel_dims.py:701`).

This is not speculative generality — YAGNI says do not build the *mechanism*, and we are not.
We are reserving six integers and two assertions, and the payoff is that the day a rented run
is justified, the diff is a YAML file rather than a refactor of the model's forward pass. The
thing that makes a model "parallelisable later" is not an abstraction layer; per
`docs/composability.md`, it is three concrete properties: a top-level forward that is mostly a
loop over child modules, layers stored in a `ModuleDict` so names survive deletion, and no
non-persistent buffers. All three are free if you do them first and expensive afterwards.

### 4.4 The checkpoint is the only Track D artifact with a cost today

`research/notes/pretraining-recipes.md` §5 already establishes the case: a 300M compute-optimal
run is `[A]` ~20.8 days locally and `[C]` ~$17–32 rented. When that trade is taken, the
checkpoint written by the local run has to load onto the rented topology. That works if and
only if the checkpoint is keyed by **parameter fully-qualified name**, with each rank
range-reading only the extents its shard covers, and resharding on load. OLMo-core does exactly
this (`training/olmo-core/src/olmo_core/distributed/checkpoint/__init__.py:702` for the save
side — `full_state_dict=False`, `cpu_offload=True`, optimizer moments staying sharded as
DTensors keyed by FQN — and `:302` for the load side, which reshards onto whatever topology is
running now). FSDP2's own state dict is sharded and requires no communication to produce
(`training/torchtitan/docs/fsdp.md:79`).

If we instead write `torch.save(model.state_dict())` now, we get a full state dict that happens
to work on one device and quietly forecloses the resharding property. That is a DR decision
disguised as a serialisation convenience. Note also the atomicity model, because it is not the
one you would design: every save is a **full rewrite** into a `<dir>-tmp` sibling followed by a
directory rename (`train/checkpoint.py:498`). No journal, no incremental delta, atomicity at
rename granularity — a torn save loses the entire checkpoint rather than a tail.

### 4.5 Gradient accumulation is our DP, and it is not numerically the same DP

We reach large global batches on one device by accumulating gradients — the mechanism is
`split_batch` (`train_module.py:393`), a spatial split by microbatch size, not a loop counter,
and `_train_microbatch_context` (`train_module.py:566`) is where a multi-rank run would
suppress the collective on all but the last microbatch. On one device the suppression is a
no-op and the accumulation is a sequential sum into `.grad`.

A DP run computes the same mathematical quantity by a **tree** reduction across ranks. These
differ in floating point. Given `[M]` that the reduction path on this machine is worth 2.8× in
relative error on a long bf16 sum (`ASSUMPTIONS.md → hipblaslt-config`), we should *expect* a
rented 8-rank run at the same seed to produce a different loss curve from the local run, and we
should write that down before it happens rather than debugging it as a bug. The honest framing:
**parallel degree is an experimental factor, and a matched-budget comparison across plans is
not automatically a matched-arithmetic comparison.**

### 4.6 The EP arithmetic is a Proteus design constraint, not just an operational one

§3.6's `k`-amplification is a modelling decision with a network price tag. Laguna's `k=10`
puts 60 KiB per token per MoE layer on the wire. If a `proteus-moe-*` arm is ever to be trained
on rented hardware, `k` is the single parameter that decides whether the run is compute-bound
or shuffle-bound, and the decision is made in the architecture file, not the launch script.
`research/notes/moe-routing-and-failure-modes.md` treats `k` as a quality/compute knob; §3.6
adds that it is also a bandwidth knob with a linear coefficient. Both are true, and only one of
them shows up in a FLOPs budget.

---

## 5. Read the code

Paths are relative to `research/reference/`. Pointers are pinned to the revisions in
`research/reference/PROVENANCE.md`.

### 5.1 The plan itself — torchtitan's `ParallelDims`

Read this file first and in full; it is the shortest complete statement of the composition
contract in any open codebase.

| Where | What to look for |
|---|---|
| `training/torchtitan/torchtitan/distributed/parallel_dims.py:181` | The product invariant: `dp_replicate · dp_shard · cp · tp · pp == world_size`. Five axes, one assertion, everything else follows. |
| `training/torchtitan/torchtitan/distributed/parallel_dims.py:273` | `fsdp = dp_shard · cp` — context parallelism is folded into the FSDP mesh, because a CP rank is also a place to put a parameter shard. |
| `training/torchtitan/torchtitan/distributed/parallel_dims.py:274` | `efsdp = fsdp · tp / ep` — EP is **carved out of** the other axes, not added. This is the line that explains why MoE models need two data-parallel meshes over one device set. |
| `training/torchtitan/torchtitan/distributed/parallel_dims.py:701` | `seq_len_divisor = tp · (cp · 2)`. The plan constrains the *data*. The factor of 2 is the causal-mask load balancer. |
| `training/torchtitan/torchtitan/distributed/parallel_dims.py:203` | The `build_mesh` docstring lists every mesh the runtime creates and which unflattening produces it. Read it as a topology map. |

### 5.2 FSDP — where the sharding granularity is decided

| Where | What to look for |
|---|---|
| `training/torchtitan/torchtitan/distributed/fsdp.py:180` | The loop that calls `fully_shard` once per transformer block. This choice — not any config field — sets the communication bucket size and therefore the peak transient memory. |
| `training/torchtitan/torchtitan/distributed/fsdp.py:281` | The root `fully_shard(model, ...)` after the per-block calls. Nested FSDP units: the root owns only the parameters no child claimed. |
| `training/torchtitan/torchtitan/distributed/fsdp.py:74` | `return not pp_enabled` — PP silently flips FSDP's resharding policy. Axis interaction in one line. |
| `training/torchtitan/torchtitan/distributed/fsdp.py:41` | `set_gradient_divide_factor(1.0)` — FSDP's built-in `1/N` is turned off because the loss is already divided by the global token count. The ragged-microbatch correctness argument, at the collective. |
| `training/torchtitan/torchtitan/distributed/fsdp.py:201` | `if efsdp_ep_size > num_experts: Shard(1)` — when there are more ranks than experts you must shard *inside* an expert rather than pad. A capacity-planning branch, in the model code. |
| `training/torchtitan/torchtitan/distributed/fsdp.py:252` | `_shard_placement_fn` returning a different mesh for expert parameters than for dense parameters. The two-mesh reality of §5.1's `efsdp` line, made executable. |
| `training/torchtitan/docs/fsdp.md:60` | The FSDP1 / FSDP2 / DeepSpeed equivalence table. The single fastest way to translate any ZeRO-stage claim in a paper into the API you are reading. |
| `training/torchtitan/docs/fsdp.md:9` | Why FSDP2 dropped `FlatParameter`: per-parameter `DTensor`s buy communication-free sharded state dicts. This is the sentence that explains why FSDP checkpoints stopped being awful. |

### 5.3 Tensor parallel — the two shardings and their reduction

| Where | What to look for |
|---|---|
| `architecture/megatron-lm/megatron/core/tensor_parallel/layers.py:869` | `ColumnParallelLinear`. The docstring gives the algebra exactly: `Y = XA + b`, `A = [A_1 … A_p]` split on its second dimension, and `gather_output` deciding whether you pay a collective here or defer it. |
| `architecture/megatron-lm/megatron/core/tensor_parallel/layers.py:1249` | `RowParallelLinear`. `A` split on the first dimension, `X` on the second — so the output is a partial sum. Read the two docstrings back to back; the pairing is the whole trick. |
| `architecture/megatron-lm/megatron/core/tensor_parallel/layers.py:525` | `LinearWithGradAccumulationAndAsyncCommunication` — the autograd function where the collective and the matmul are interleaved for overlap. This is where "bytes" becomes "time". |
| `training/torchtitan/torchtitan/models/common/decoder_sharding.py:65` | `colwise_config()` — weight `Shard(0)`, output `Shard(-1)`. |
| `training/torchtitan/torchtitan/models/common/decoder_sharding.py:91` | `out_src_shardings=dense_activation_placement(tp=spmd.P)` — **Partial**. The output of a row-parallel linear is not a value, it is an unreduced promise. If you read one line in this file, read this one. |
| `training/torchtitan/torchtitan/models/common/decoder_sharding.py:209` | `q_placements = dense_activation_placement(tp=spmd.S(2))` — attention is TP-sharded on the *head* dimension. This is where §4.1's KV-cache head-sharding comes from. |
| `training/torchtitan/torchtitan/models/common/decoder_sharding.py:53` | `dense_sequence_parallel_placement()` with `partition_spec=(DP, (CP, TP), None)` — sequence parallelism as a declarative placement, and note that CP and TP jointly shard the sequence axis. |

### 5.4 Pipeline parallel — schedule, split, and the naming problem

| Where | What to look for |
|---|---|
| `architecture/megatron-lm/megatron/core/pipeline_parallel/schedules.py:922` | `num_warmup_microbatches = pp_size − pp_rank − 1`. Plain 1F1B in one line: earlier stages run more forwards before their first backward. |
| `architecture/megatron-lm/megatron/core/pipeline_parallel/schedules.py:929` | The interleaved version: `(pp_size − pp_rank − 1)·2 + (chunks − 1)·group_size`. Diff these two lines and you have the entire interleaving idea. |
| `architecture/megatron-lm/megatron/core/pipeline_parallel/schedules.py:2127` | `forward_backward_pipelining_without_interleaving` — the reference 1F1B loop, with its send/recv placement. |
| `training/torchtitan/torchtitan/distributed/pipeline_parallel.py:263` | The warning when microbatches < total stages. The bubble formula, as a runtime check. |
| `training/torchtitan/torchtitan/distributed/pipeline_parallel.py:446` | `model = copy.deepcopy(whole_model)` then delete. The splitter is blunt on purpose: stage membership is a *name* problem. |
| `training/torchtitan/torchtitan/distributed/pipeline_parallel.py:517` | `pp_rank + s · pp_degree` — the loop-style virtual-stage assignment. Compare with the `v`-style branch two lines down. |
| `training/torchtitan/torchtitan/distributed/pipeline_parallel.py:164` | `input_weight` / `output_weight` — embeddings and the LM head are counted as multiple layers when balancing stages, because they are not the same cost as a transformer block. |
| `training/torchtitan/docs/composability.md:14` | Why `ModuleDict` and not `ModuleList`: deleting `layers.0` must leave `layers.1` named `layers.1`, or distributed checkpointing loses its identifiers. |
| `training/torchtitan/docs/composability.md:19` | The seed checkpoint. Read this as an admission: there is no clean way to shard an RNG stream, so they initialise centrally and load. |
| `training/olmo-core/src/olmo_core/distributed/parallel/pipeline_parallel.py:30` | A one-line honesty note in a docstring: the zero-bubble variants "have several issues at the moment". The frontier, as recorded by people shipping it. |

### 5.5 Expert parallel — the data-dependent collective

| Where | What to look for |
|---|---|
| `training/torchtitan/torchtitan/models/common/token_dispatcher.py:273` | The *counts* all-to-all — a collective whose only job is to tell you the shape of the next collective. |
| `training/torchtitan/torchtitan/models/common/token_dispatcher.py:313` | `# NOTE: this would incur a device-to-host sync`. A mandatory host round trip inside the forward pass, once per MoE layer. |
| `training/torchtitan/torchtitan/models/common/token_dispatcher.py:340` | The dispatch `all_to_all_single` with variable split sizes. |
| `training/torchtitan/torchtitan/models/common/token_dispatcher.py:368` | The combine, with `input_splits` and `output_splits` swapped. Read them together to see the symmetry. |
| `training/torchtitan/torchtitan/models/common/token_dispatcher.py:93` | `_local_reorder` — the argsort that groups tokens by expert before the shuffle, so the all-to-all can be one contiguous send per destination. This is a sort-before-shuffle, exactly as in MapReduce. |
| `training/torchtitan/torchtitan/distributed/fsdp.py:297` | The comment explaining why EP forces *explicit* FSDP prefetching: the D2H syncs from `:313` break the implicit heuristic. Two subsystems interfering through a third. |
| `training/torchtitan/torchtitan/distributed/activation_checkpoint.py:53` | `torch.ops.aten.topk.default` on the must-save list, "to keep MoE expert assignments stable between forward and recompute". |
| `training/torchtitan/torchtitan/distributed/activation_checkpoint.py:62` | Collectives on the must-save list so recompute does not re-communicate. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:185` | Laguna's `e_score_correction_bias` — load balancing as a learned bias on *selection* only, leaving combination weights untouched. The systems problem solved in the objective. |

### 5.6 Data parallel, the DDP baseline, and bucketing

| Where | What to look for |
|---|---|
| `architecture/megatron-lm/megatron/core/distributed/param_and_grad_buffer.py:92` | `_ParamAndGradBucket` — the coalescing unit. Write-combining for collectives. |
| `architecture/megatron-lm/megatron/core/distributed/param_and_grad_buffer.py:979` | The bucketing rule: close a bucket once it exceeds `bucket_size` elements. The α/β trade of §3.2, as a loop. |
| `architecture/megatron-lm/megatron/core/distributed/param_and_grad_buffer.py:594` | `start_grad_sync` — where overlap actually happens, and where the "should not have multiple communication calls outstanding" assertion lives. |
| `training/olmo-core/src/olmo_core/distributed/parallel/data_parallel.py:49` | HSDP's default: one replica per node, shard degree = GPUs per node. §3.7's ordering heuristic, as a default. |
| `training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:566` | `_train_microbatch_context` — the coalescing point where gradient reduction is suppressed on all but the last microbatch. On one device this is a no-op, which is exactly why it is worth reading now. |

### 5.7 Context parallel, because it is the one that touches the KV cache

| Where | What to look for |
|---|---|
| `training/torchtitan/torchtitan/distributed/context_parallel.py:222` | `_HeadTailLoadBalancer` — each rank gets one chunk from the front and one from the back, because causal masking makes the workload monotone in position. Range partitioning with a balancer baked in. |
| `training/torchtitan/torchtitan/distributed/context_parallel.py:73` | `flex_cp_allgather(k, v, ...)` — in this (explicitly temporary) path, K and V are **all-gathered** so the kernel sees full-length keys. Read this and then ask what CP actually saves: query-side work and activation memory, not KV residency. |
| `training/torchtitan/torchtitan/models/common/decoder_sharding.py:211` | `kv_dst_placements = ...(tp=S(2), cp=R)` — K and V are Replicate on the CP axis while Q stays sharded. The same fact, stated declaratively. |

---

## 6. Exercises

All of these are CPU-only by construction. There is no GPU variant, because the GPU cannot do
the thing. Exercise A needs the lab venv; A′ needs the non-lab interpreter; B and C need
nothing but a Python interpreter and a pen.

Timings below are wall clock measured while writing this module, on the Z13. Note that on this
machine `import torch` from the ROCm wheel is itself a two-to-three-minute operation from cold,
which dominates Exercise A entirely.

### Exercise A — establish the blast radius of "no collectives"

**Difficulty: easy. Time: 5 minutes, of which ~3 is `import torch`.**

Right now `ASSUMPTIONS.md → single-device-only` carries a `[C]` tag citing ROCm documentation.
Turn it into an `[M]` and find out whether the cited mechanism is even the right one.

Write a probe that, in a **fresh process**, prints: `torch.__version__`,
`torch.version.hip`, `torch.distributed.is_available()`, the result of each
`torch.distributed.is_*_available()` accessor, whether `torch._C._distributed_c10d` imports,
and whether each of `torch.distributed.tensor`, `torch.distributed.fsdp`,
`torch.distributed.pipelining` imports. Run it twice, in two separate processes, because a
single observation is an anecdote.

Then count how much of a real training framework this rules out:

```
rg -c "^\s*(import torch\.distributed|from torch\.distributed)" --glob "*.py" \
   research/reference/training/torchtitan/torchtitan
```

**Reference results** (lab venv `C:\venvs\lab`, `torch 2.12.0a0+rocm7.13.0a20260313`,
HIP 7.2.0, native Windows, gfx1151, 2026-07-26, two fresh processes, identical output both
times):

| Probe | Result |
|---|---|
| `torch.distributed.is_available()` | **False** |
| `torch.distributed.is_gloo_available` | attribute **absent** (not False — absent) |
| `is_nccl_available` / `is_mpi_available` / `is_ucc_available` / `is_xccl_available` | absent |
| `import torch._C._distributed_c10d` | `ModuleNotFoundError: No module named 'torch._C._distributed_c10d'; 'torch._C' is not a package` |
| `import torch.distributed.tensor` | same `ModuleNotFoundError` |
| `import torch.distributed.fsdp` | same |
| `import torch.distributed.pipelining` | same |
| `dist.init_process_group` | `AttributeError: module 'torch.distributed' has no attribute 'init_process_group'` |
| torchtitan files with a `torch.distributed` import | **83 files, 188 occurrences** |

**The number to check: 83.** Eighty-three of torchtitan's own modules cannot be imported in
the lab venv. Not "cannot run" — cannot *import*. That is the honest measure of how much of the
training-systems world is closed to this machine, and it is the reason this module's remaining
exercises are arithmetic.

**What it changes.** `ASSUMPTIONS.md` and `research/memory/open-problems-ranked.md` both state
the constraint as "collectives are incomplete", citing `torch._C._distributed_c10d` being
incomplete on gfx1151. The measurement says the module is **absent from the build entirely**,
which is a different and larger claim: it removes single-device DTensor, CPU meshes, placement
assertions, and the ability to unit-test any of §4's layout code in the lab venv. Propose the
row update; do not edit the record.

**Exercise A′ — the optional half, and it is worth the extra half hour.** `[M]` The *non-lab*
interpreter on this machine — `C:\Users\solar\AppData\Local\Programs\Python\Python312\
python.exe`, `torch 2.11.0+cu128` — **does** have `torch.distributed` with the Gloo backend, and
FSDP2 runs there on CPU. Which means the one thing this module says you cannot do, you can
actually do once, small, for semantics.

Build a four-block `nn.Linear(64, 64, bias=False)` stack, `fully_shard` each block on a
`("cpu", (2,))` mesh, spawn two Gloo ranks with `torch.multiprocessing.spawn` and a
file-based store, and run one forward and one backward under a `TorchDispatchMode` that counts
any op whose qualified name contains `c10d`.

**Reference results** (2026-07-26, world_size=2, CPU, fp32):

```
full param numel        16384
local (sharded) numel    8192        ratio 2.00
param type              DTensor      placements (Shard(dim=0),)
global shape (64, 64)                local shape (32, 64)
grad type               DTensor      placements (Shard(dim=0),)   local shape (32, 64)

collectives in ONE forward + backward:
    8 x c10d._allgather_base_.default
    4 x c10d._reduce_scatter_base_.default
```

**The two numbers to check: 8 and 4.** Four FSDP units, two all-gathers each (one in forward,
one in backward, because `reshard_after_forward` defaults to True outside PP) and one
reduce-scatter each. That is §3.3's `(2·b_p + b_r)·P` formula, observed as op counts rather
than derived. The root `fully_shard(model)` contributes nothing because every parameter was
already claimed by a block — which is the nesting rule from `training/torchtitan/docs/fsdp.md:50`
made visible.

**Then make one prediction and test it**, which is the actual point of the exercise: pass
`reshard_after_forward=False` to each block and predict the new counts before you run. The
formula says the backward all-gather disappears, so **4 and 4**. I did not run that variant —
it is left as the falsifiable half, and if you get something other than 4 and 4 the interesting
work is explaining why, not correcting the number.

A trap inside the trap, worth hitting once: if you also sum `args[0].numel() * element_size()`
you will get 163,840 bytes and it will not match your hand calculation of 131,072. The reason
is that the in-place `_base_` variants take the **output** buffer as the first argument, so for
all-gather you have counted the gathered tensor (full size) and for reduce-scatter the
scattered one (local size). Fix the indexing and the arithmetic closes. Instrumenting a
collective is not the same as instrumenting a function call, and this is the cheapest possible
demonstration of that.

**Now the trap around the whole exercise.** That interpreter is **not the lab instrument**. It
is a CUDA build with no usable GPU, and `CLAUDE.md → torch-build` explicitly rules it out of the
lab. Nothing measured there is evidence about our hardware, our numerics, or our throughput —
the numbers above are statements about *PyTorch's semantics*, which are portable, not about
gfx1151, which is not. Use it as an executable documentation reader, tag nothing `[M]` about
the Z13 from it, and keep it out of `uv sync`.

### Exercise B — the communication-volume calculator

**Difficulty: medium (the derivations), easy (the code). Time: 60–90 minutes with a pen,
then 20 minutes of typing. Runtime under 1 second.**

Do the algebra first, on paper, before writing any code. Derive, for `N` ranks:

1. per-rank resident bytes per parameter for DDP, ZeRO-1, ZeRO-2, ZeRO-3;
2. FSDP and DDP bytes per rank per step as functions of `b_p` and `b_r`;
3. TP bytes per rank per step;
4. the token count at which TP volume equals FSDP volume;
5. PP bytes per seam per step, and the bubble fraction;
6. EP bytes per rank per step.

Then write ~60 lines of **pure Python — no torch** implementing them, and evaluate at the
reference arm: `L=24`, `d=1024`, `V=50,257`, FFN `4d`, tied embeddings; and at Laguna S 2.1:
`L=48`, `d=3072`, `E=256`, `k=10`, and — read this out of `config.json`, do not assume it —
`L_moe = 47`, because `mlp_only_layers` is `[0]`.

**Reference outputs.** These are arithmetic from the stated model, not hardware measurements;
they should reproduce exactly.

```
params per block                12,582,912
non-embedding P                301,989,888
total P (tied embeddings)      353,453,056

per-rank memory, N=8
  DDP    16.00 B/param   5.267 GiB
  ZeRO-1  5.50 B/param   1.810 GiB
  ZeRO-2  3.75 B/param   1.234 GiB
  ZeRO-3  2.00 B/param   674.158 MiB
  + transient all-gather buffer, one block bf16   24.000 MiB  (48.000 MiB with prefetch)

per-step comm bytes per rank, N=8
  b_r = 4 (fp32, torchtitan default)
    DDP 2.304 GiB | FSDP reshard=True 2.304 GiB (1.000x) | reshard=False 1.728 GiB (0.750x)
  b_r = 2 (bf16)
    DDP 1.152 GiB | FSDP reshard=True 1.728 GiB (1.500x) | reshard=False 1.152 GiB (1.000x)

TP vs FSDP, 65,536 tokens/step
  TP(8)   21.000 GiB/rank/step
  FSDP(8)  2.304 GiB/rank/step
  crossover                     7,191 tokens/step

PP
  p=4 m=8   bubble 0.273   (interleaved v=2: 0.158)
  p=4 m=32  bubble 0.086   (v=2: 0.045)
  p=8 m=8   bubble 0.467   (v=2: 0.304)
  p=8 m=64  bubble 0.099   (v=2: 0.052)
  p2p bytes, p=4, 65,536 tokens   768.000 MiB
  TP(8) : PP(4) volume ratio      28.0x

Laguna S 2.1, 8,192 tokens/rank/step, L_moe = 47
  EP=8    77.109 GiB/rank/step
  EP=16   82.617 GiB/rank/step
  EP=64   86.748 GiB/rank/step
  one dispatch all-to-all at EP=64      472.500 MiB
  per MoE layer at EP=64                  1.845 GiB   (4 all-to-alls)
  wire bytes per token per MoE layer     60.000 KiB   (hidden state: 6.000 KiB -> 10x)
```

**The three numbers to check: 1.000, 7,191, 10×.**

- **1.000** is the FSDP:DDP byte ratio at the shipped dtypes. If you get 1.5 you used
  `b_r = b_p`; go back and read `configs.py:66`.
- **7,191** is the TP/FSDP crossover in tokens per step. It is far below any realistic batch,
  which is the quantitative form of "TP is the expensive axis."
- **10×** is the EP wire amplification for `k=10`. Notice that `E=256` does not appear in the
  formula at all — the number of experts is irrelevant to the volume; only `k` matters. That
  surprises most people and it is worth confirming symbolically before you accept it.

**A question to answer in one sentence before moving on:** why does `EP=64` move only 12.5%
more than `EP=8` when the device count grew 8×?

### Exercise C — plan search at `world_size = 8`, and the rent decision

**Difficulty: medium. Time: 30–45 minutes. Runtime under 1 second.**

Extend Exercise B into a search. Enumerate every 5-tuple
`(dp_replicate, dp_shard, cp, tp, pp)` of powers of two whose product is 8, discard any that
violates `seq_len % (tp · cp · 2) == 0` at `seq_len = 8192`, and score each by total bytes per
rank per step at 65,536 tokens per step, remembering that:

- FSDP's group is `dp_shard · cp` and it shards `P / pp` parameters (one stage's worth);
- the replicate axis all-reduces `P / (pp · dp_shard · cp)` parameters;
- TP and PP see `65,536 / (dp_replicate · dp_shard)` tokens;
- TP's layer count is `L / pp`.

Then divide by a link bandwidth and compare against a step time.

**Reference outputs:**

```
valid plans: 35

best six:
  dp_rep=1 dp_shard=2 cp=1 tp=1 pp=4   ->  721.079 MiB / rank / step
  dp_rep=2 dp_shard=1 cp=1 tp=1 pp=4   ->  721.079 MiB          (tie)
  dp_rep=1 dp_shard=4 cp=1 tp=1 pp=2   ->    1.050 GiB
  dp_rep=2 dp_shard=2 cp=1 tp=1 pp=2   ->    1.050 GiB          (tie)
  dp_rep=4 dp_shard=1 cp=1 tp=1 pp=2   ->    1.050 GiB          (tie)
  dp_rep=1 dp_shard=1 cp=2 tp=1 pp=4   ->    1.079 GiB
worst three:
  dp_rep=1 dp_shard=1 cp=4 tp=2 pp=1   ->   13.975 GiB
  dp_rep=1 dp_shard=1 cp=2 tp=4 pp=1   ->   19.317 GiB
  dp_rep=1 dp_shard=1 cp=1 tp=8 pp=1   ->   21.000 GiB

spread best:worst = 21504 / 721.079 = 29.8x

best plan at 900 GB/s (NVLink-class)      0.84 ms/step
best plan at  50 GB/s (fast Ethernet)    15.12 ms/step
worst plan at 900 GB/s                   25.05 ms/step
worst plan at  50 GB/s                  450.97 ms/step
```

**The three numbers to check: 35, 29.8×, and the fact that the winner is `pp=4`.**

- **35** you can derive without code. The number of ordered 5-tuples of powers of two with
  product `2^w` is `C(w+4, 4)`; at `w = 3` that is `C(7,4) = 35`. Confirm your enumeration
  against the closed form — if they disagree, your enumeration has a bug, and this is the
  cheapest possible test of it.
- **29.8×** is the spread between the best and worst plan on the same hardware for the same
  model. Plan choice is a 30× decision on communication, which is why §8 lists automatic
  planning as an open problem rather than a solved one.
- **`pp=4` wins, and that is the byte model lying to you.** Now write down, in your own words,
  the three things the score function cannot see: the pipeline bubble (at `p=4` and `m=8`,
  27.3% of the step, per Exercise B), the activation memory that `pp=4, dp=2` concentrates on
  stage 0, and the fact that FSDP's all-gather overlaps with compute while TP's all-reduce
  largely does not. Then re-rank the top five plans by hand with those in mind. **The exercise
  is the disagreement between the two rankings**, not either ranking.

**The rent tie-in.** `research/notes/pretraining-recipes.md` §5 puts a 300M compute-optimal run
at `[A]` ~20.8 days locally and `[C]` ~$17–32 on one rented H100. Using the step time from
§3.4 (~49.6 ms of compute per rank at 350 TFLOP/s on 8 ranks) and the best plan's 0.84 ms of
communication, compute the communication overhead as a percentage of step time on an
NVLink-class fabric and on a 50 GB/s fabric. **You should get roughly 1.7% and 30%.** That
single pair of numbers is the argument for renting a single well-connected node rather than
eight cheap ones, and it is the form the G3 unit-economics question takes when we eventually
ask it.

---

## 7. Self-check

Answers at the end of the module.

1. FSDP shards parameters, gradients and optimizer state; DDP replicates all three. Why, at
   torchtitan's shipped dtypes, do they move the *same* number of bytes per step — and what
   would you have to change to recover the textbook 1.5× ratio?

2. You have 8 GPUs in one node on NVLink and a 300M-parameter dense model with a 65,536-token
   global batch. Your colleague proposes `tp=8`. Give the two quantitative reasons this is
   wrong and the one situation in which it would be right.

3. Expert parallelism with `E = 256` experts and `k = 10` moves ten copies of each token's
   hidden state per MoE layer. If you doubled `E` to 512 while holding `k = 10`, what happens
   to the all-to-all volume, and why?

4. Pipeline parallelism moves 28× fewer bytes than tensor parallelism for the same model and
   batch. Why, then, is PP not simply the default everywhere — name the two costs the byte
   model cannot see.

5. Selective activation checkpointing recomputes the forward pass during backward to save
   memory. Name two things torchtitan explicitly refuses to recompute, and explain what breaks
   if you recompute each of them.

6. `ASSUMPTIONS.md → hipblaslt-config` records a 2.8× change in bf16 long-reduction error from
   a library configuration flag. What does that imply about comparing an experimental arm run
   at `tp=1` against the same arm at `tp=8`, and what would you have to do to make that
   comparison legitimate?

---

## 8. What is still unsolved here

**1. Nobody knows how to choose a plan.** `[C]` Alpa (2201.12023) automated inter- and
intra-operator parallelism search in 2022; no frontier lab uses an automatic planner for
flagship runs. Plans are hand-tuned, and the tuning knowledge is folklore. The 2026 literature
is still attacking it from several directions — `[C]` WLB-LLM (2503.17924) on workload
imbalance across PP and CP; `[C]` CFP (2504.00598) on intra-operator plan optimisation; `[C]`
Tangram (2606.16907) on making heterogeneity-unaware planners work on mixed GPU clusters; `[C]`
AutoSP (2604.27089) on compiler-driven sequence parallelism for long context. That this many
groups are still publishing search methods is the evidence that none has won. Our Exercise C is
a toy version of the problem and it already shows a 29.8× spread over 35 candidates; the real
space at 1024 devices has 1001 power-of-two plans before you add schedules, dtypes and
checkpointing policies.

**2. "Zero bubble" is contested once you count everything.** `[C]` Zero Bubble (2401.10241)
achieves a genuinely empty pipeline under synchronous semantics by splitting the backward pass
and deferring weight-gradient computation — at roughly twice the activation memory of the
interleaved schedule and requiring at least `2p−1` microbatches. Whether that is "negligible in
large clusters where memory is abundant" is precisely the disputed premise. OLMo-core's own
docstring says the zero-bubble variants "have several issues at the moment"
(`training/olmo-core/src/olmo_core/distributed/parallel/pipeline_parallel.py:30`), which is a
shipping team's assessment against a paper's claim.
The 2026 direction is to buy the bubble back by relaxing synchrony — `[C]` "Breaking the
Bubble: Asynchronous Pipeline Parallel Training with Bounded Weight Inconsistency"
(2606.07881) — which reopens the async-SGD question the field closed in 2018. `[C]` See also
SlimPipe (2504.14519) and PipeOffload (2503.01328) on the memory side, and `[C]` 2605.24006 (a
tabular schedule abstraction for communication-aware comparison of PP schedules, May 2026),
which exists because *comparing* schedules is itself unsolved.

**3. All-to-all is the MoE bottleneck and there is no accepted fix.** Reported at 34–40% of
step time `[C]` (MegaScale-MoE, 2505.11432; DisagMoE, 2605.11005). The approaches in flight —
hierarchical all-gather-then-local-all-to-all, computation/communication overlap, disaggregating
attention from FFN onto separate GPU groups, expert-collaboration-aware placement `[C]` (Occult,
2505.13345), and DeepEP's dedicated kernels (a GitHub project, not a paper; torchtitan has an
integration at `torchtitan/distributed/deepep/`) — do not agree on a winner. Underneath the
systems problem sits an unsolved *joint* problem: expert load balance is controlled by the
training objective (an auxiliary loss, or Laguna's learned selection bias) while the tail
latency it causes is a systems cost, and no one optimises both together.

**4. Low-communication training might make the whole 4D plan less important.** `[C]` DiLoCo
(2311.08105) trains with inner optimisation steps between rare outer synchronisations; `[C]`
Streaming DiLoCo (2501.18512) overlaps the outer communication; `[C]` SparseLoCo (2508.15706)
and `[C]` HeLoCo (2606.00271, heterogeneous and asynchronous) continue the line. If these hold
at frontier scale the interconnect stops being the design centre. **Contested:** no frontier lab
has published a flagship model trained this way, and the published scaling studies are small
relative to the models they aim to displace. Treat as promising and unproven.

**5. Availability decreases with scale and there is no accepted answer.** Fault-tolerant
training — elastic membership, hot spares, partial-replica recovery — exists as research and as
experimental code (torchtitan carries a `torchft` integration under `experiments/`, which is
where torchtitan puts things it does not yet stand behind). The production answer remains
frequent checkpointing plus job restart, which is a DR strategy with an RPO measured in
minutes of GPU-hours.

**6. Nobody publishes numerics equivalence across plans.** This is the gap that should worry a
lab whose results are supposed to mean something. Changing TP degree changes a floating-point
reduction tree; changing DP degree changes the gradient reduction tree; changing PP degree
changes nothing arithmetically but changes what is recomputed and therefore which non-
deterministic ops re-run. The strongest statement in the ecosystem is torchtitan's contributor
instruction to "verify numerics match before and after across multiple parallelism configs" —
an instruction, not a result. `[M]` We have local evidence that this machine's reduction path is
worth 2.8× in relative error on a long bf16 sum. So for us the open question is sharp: **is a
result obtained at one parallel plan transferable to another, and by how much?** No one has
published the answer at the precision an ablation rig needs.

**7. Whether gfx1151 will ever ship `c10d` on Windows is unknown, and the fallback has its own
defect.** `[M]` `c10d` is absent from the current native-Windows nightly. WSL2 is `[C]` now
officially supported for this SKU via ROCDXG (`CLAUDE.md → ENVIRONMENT`), and a Linux wheel
would normally carry `c10d` and RCCL — but **we have not tested whether it does on gfx1151, and
we should not assume it**; every capability claim about this platform that we have actually
checked has come back different from the documentation. What is documented is that WSL2 `[C]`
clamps the ROCm pool to the `.wslconfig` value (ROCm #6022), destroying the ≥62 GiB fast tier
that justifies this machine at all. So the two local options appear to be: no collectives with
62 GiB, or possibly-collectives with far less — and even one GPU makes the whole question moot,
since every collective in this module needs at least two. The honest plan is to rent when a
specific result justifies it, and to spend local wall clock on the capacity-bound experiments
this machine is uniquely good at.

**8. The memory-track-specific gap, which is ours to take.** There is no published
characterisation of how KV-cache management behaves under a parallel plan: eviction policies
under head-sharding (TP), under non-contiguous position-sharding (CP), or under layer-sharding
(PP). Every eviction paper we surveyed in `research/memory/kv-compression-and-eviction.md`
evaluates single-device. The design-time half of that question — what a cache-layout descriptor
must express so a policy means the same thing at any plan — costs nothing to answer now and is
§4.1's subject. The measurement half needs hardware we do not have.

---

## Answers to the self-check

**1.** Because the byte counts are `(2·b_p + b_r)·P·(N−1)/N` for FSDP with
`reshard_after_forward=True`, and `2·b_r·P·(N−1)/N` for DDP. torchtitan ships `b_p = 2` (bf16
parameters) and `b_r = 4` (fp32 reduction, and the type annotation at `configs.py:66` is
`Literal["float32"]`, so it is not even configurable): `2·2 + 4 = 8` and `2·4 = 8`. Identical —
2.304 GiB per rank per step at the reference arm on 8 ranks. To recover 1.5× you would have to
reduce gradients in bf16, at which point DDP moves 1.152 GiB and FSDP 1.728 GiB. The
substantive point is that FSDP's win at the shipped defaults is *entirely* memory — 674 MiB
resident versus 5.267 GiB — at bandwidth parity, and the widely-quoted 1.5× penalty does not
exist in that configuration.

**2.** Reason one: volume. TP-8 moves 21.000 GiB per rank per step against FSDP-8's
2.304 GiB — 9.1× more — because TP's volume scales with tokens (`L·4·2·T·d·b·(N−1)/N`) while
FSDP's scales with parameters, and the crossover is at only 7,191 tokens per step. Reason two:
overlap. TP's all-reduce sits on the critical path of every layer — the next operation consumes
its output — whereas FSDP's all-gather for block `i+1` runs under block `i`'s compute. At
900 GB/s the TP traffic alone is 25.0 ms against roughly 49.6 ms of compute, i.e. ~50%
overhead, mostly unhidden. It would be right if the model did not fit any other way: TP is the
axis that reduces *per-layer* parameter and activation footprint simultaneously, so for a model
whose single layer exceeds one device it is the only option. At 300M parameters and 62 GiB per
device, that situation does not arise.

**3.** Nothing happens. The volume is `L_moe · 4 · T · k · d · b_p · (ep−1)/ep` and `E` does not
appear. Each token sends its hidden state to `k` experts regardless of how many exist; `E`
changes the *parameter* count (and therefore FSDP/EFSDP traffic and memory) but not the shuffle.
The corollary is the design lever: to cut MoE communication you must cut `k`, not `E` — and
cutting `k` changes model quality while cutting `E` changes capacity. The related surprise from
Exercise B is that raising `ep` from 8 to 64 adds only 12.5%, because the `(ep−1)/ep` factor is
already 0.875 at `ep=8` and saturates at 1.

**4.** Cost one: the bubble. The pipeline must fill and drain once per optimizer step because
the semantics are synchronous — `(p−1)/(v·m + p−1)`, which at `p=4, m=8, v=1` is 27.3% of the
step, idle. Bytes are cheap; time is not. Cost two: activation retention. Every microbatch's
forward activations must stay resident until its backward pass reaches that stage, so stage 0
holds the most; GPipe holds `m` microbatches' worth and 1F1B caps it at `p`, which is the entire
reason 1F1B replaced GPipe despite having an identical bubble ratio. A third, non-numeric cost:
stage boundaries must fall on module boundaries that preserve parameter names for checkpointing
(`docs/composability.md:14`), and initialisation needs a centrally-generated seed checkpoint
(`:19`) because RNG order is a global side effect.

**5.** (a) `torch.ops.aten.topk.default` (`activation_checkpoint.py:53`) — topk can be
non-deterministic, so recomputing it can assign a token to a *different* expert than the
forward pass used. The backward pass would then compute gradients for experts that never
processed that token. Nothing crashes; the model just trains slightly wrong, forever, and no
metric shows it. (b) the outputs of `reduce_scatter_tensor` and `all_to_all_single`
(`activation_checkpoint.py:62`, `:63`) — recomputing a collective means *re-issuing* it, which
at best doubles the communication and at worst deadlocks, because the peer ranks are not
expecting a second call at that point in the program. Both are cases where a purely local
memory optimisation has a non-local consequence.

**6.** It implies that the two runs are not the same experiment. TP-8's row-parallel linear
produces a floating-point sum over 8 partial products where TP-1 produces a single dot product;
that is a reassociation of the same sum, and reassociation of floating-point addition is not
identity-preserving. Our own measurement — 2.01e-3 versus 5.60e-3 relative error on a
length-1,048,576 bf16 weighted sum, purely from a change in how the reduction is performed —
puts a number on how large such an effect can be on this hardware. To make the comparison
legitimate you would have to either (a) hold the plan fixed across all arms of the ablation and
report the plan as part of the configuration, which is the cheap and correct answer, or (b) run
a plan-only control — same arm, same seed, two plans — and show the difference is inside the
seed-to-seed band before attributing anything to the arm. Option (b) is the experiment §8 item
6 says nobody has published.

---

## Sources

### Local measurements produced for this module

- **Distributed-surface probe.** Lab venv `C:\venvs\lab`, `torch 2.12.0a0+rocm7.13.0a20260313`
  (HIP 7.2.0), native Windows, gfx1151, 2026-07-26. Two independent fresh processes, identical
  output: `torch.distributed.is_available() == False`; the `is_*_available` accessors are
  absent from the module namespace; `torch._C._distributed_c10d` raises
  `ModuleNotFoundError: ... 'torch._C' is not a package`; `torch.distributed.tensor`,
  `torch.distributed.fsdp` and `torch.distributed.pipelining` all fail with the same error;
  `dist.init_process_group` is absent. Tagged `[M]`. This is a measurement of the *software
  instrument*, not of the GPU; no seeds are involved and the result is deterministic.
- **torchtitan import census.** `rg -c "^\s*(import torch\.distributed|from torch\.distributed)"
  --glob "*.py"` over `research/reference/training/torchtitan/torchtitan`: **83 files, 188
  occurrences**, 2026-07-26, at the revision in `PROVENANCE.md`.
- **Non-lab interpreter, CPU only.** `C:\Users\solar\AppData\Local\Programs\Python\Python312\
  python.exe`, `torch 2.11.0+cu128`, 2026-07-26. One process at world_size 1:
  `dist.is_available() == True`, Gloo available, NCCL not; a one-rank CPU `DeviceMesh` plus
  `fully_shard` produces parameters of type `DTensor` with placements `(Shard(dim=0),)` and
  completes forward and backward. Two processes at world_size 2 via
  `torch.multiprocessing.spawn` and a file store, four `nn.Linear(64,64,bias=False)` blocks
  each its own FSDP unit, fp32: local shard is exactly half (8192 of 16384 numel; local shape
  `(32, 64)` of global `(64, 64)`), gradients are `DTensor` with the same placement, and one
  forward-plus-backward emits **8 `c10d._allgather_base_.default` and 4
  `c10d._reduce_scatter_base_.default`** — two all-gathers and one reduce-scatter per unit,
  which is §3.3's formula counted rather than derived. Recorded because it is the only way to
  execute FSDP2 semantics on this machine. **Explicitly not the lab instrument**
  (`CLAUDE.md → torch-build`); these are statements about PyTorch semantics, which are
  portable, and no claim about gfx1151, throughput or numerics is derived from them.
- **Communication arithmetic.** All byte counts, ratios, crossovers, bubble fractions and the
  35-plan enumeration in §3 and §6 are deterministic arithmetic from the cost model stated in
  §3.1–§3.2, computed 2026-07-26 and reproducible by Exercise B/C. They are **not**
  measurements of any interconnect, and they are labelled as arithmetic throughout rather than
  `[M]`.

### Repo `[M]` inputs used but not re-measured here

`ASSUMPTIONS.md` rows `gpu-fast-tier-size` (≥62 GiB at ~200 GB/s, single run per arm),
`gemm-throughput-below-reference` (20.9 TFLOP/s bf16 at 8192³),
`hipblaslt-config` (bf16 long-reduction relative error 2.01e-3 configured vs 5.60e-3 unset,
N=1,048,576, 3 seeds, fresh processes — the 2.8× that makes reduction order a numerics axis),
`large-tensor-fault-32gib` (silent hang at 0 CPU — the failure signature a deadlocked collective
would also present),
`reference-model` and `laguna-heads-uniform` and `kv-per-token-laguna` (Laguna S 2.1 config at
revision `b0a9fd7c850e`: 48 layers, `hidden_size` 3072, `num_experts` 256,
`num_experts_per_tok` 10, `num_key_value_heads` 8 uniform),
`torch-build`, `single-device-only`, `bf16-numerics-unproven`.
`research/notes/pretraining-recipes.md` §5 (wall-clock ladder and rented-H100 pricing).

### Code pointers

Every `file:line` in §3 and §5 was opened and the named construct confirmed on the named line
on 2026-07-26, against the revisions in `research/reference/PROVENANCE.md`.

Introduced by this module:
`training/torchtitan/torchtitan/distributed/parallel_dims.py:181`, `:203`, `:273`, `:274`,
`:701`;
`training/torchtitan/torchtitan/distributed/fsdp.py:41`, `:74`, `:147`, `:180`, `:201`, `:252`,
`:281`, `:297`, `:299`;
`training/torchtitan/torchtitan/distributed/pipeline_parallel.py:164`, `:263`, `:446`, `:517`;
`training/torchtitan/torchtitan/distributed/context_parallel.py:73`, `:222`;
`training/torchtitan/torchtitan/distributed/activation_checkpoint.py:53`, `:62`, `:63`;
`training/torchtitan/torchtitan/models/common/decoder_sharding.py:53`, `:65`, `:91`, `:209`,
`:211`;
`training/torchtitan/torchtitan/models/common/token_dispatcher.py:93`, `:273`, `:313`, `:340`,
`:368`;
`training/torchtitan/torchtitan/config/configs.py:66`;
`training/torchtitan/docs/fsdp.md:9`, `:50`, `:60`, `:79`;
`training/torchtitan/docs/composability.md:14`, `:19`;
`architecture/megatron-lm/megatron/core/tensor_parallel/layers.py:525`, `:869`, `:1249`;
`architecture/megatron-lm/megatron/core/pipeline_parallel/schedules.py:922`, `:929`, `:2127`;
`architecture/megatron-lm/megatron/core/distributed/param_and_grad_buffer.py:92`, `:594`,
`:979`;
`training/olmo-core/src/olmo_core/distributed/parallel/data_parallel.py:49`;
`training/olmo-core/src/olmo_core/distributed/parallel/pipeline_parallel.py:30`.

Reused from `research/reference/CODE_MAP.md` (machine-verified by
`scripts/generate_code_map.py`):
`training/olmo-core/src/olmo_core/distributed/checkpoint/__init__.py:302`, `:702`;
`training/olmo-core/src/olmo_core/train/checkpoint.py:498`;
`training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:393`, `:566`;
`architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:183`, `:185`.

Laguna pointers introduced by this module, into the *model artifact* copy (a different file
from the `transformers` copy above, with different line numbers — check which one a pointer
names before following it): `models/laguna-s/modeling_laguna.py:246` (the always-on shared
expert), `:480` (the `mlp_only_layers` / `decoder_sparse_step` branch that makes layer 0 dense),
and `models/laguna-s/config.json:28` (`mlp_only_layers: [0]`), `:110` (`mlp_layer_types`).

### arXiv `[C]`

**Foundations of each axis.**
`2006.15704` — *PyTorch Distributed: Experiences on Accelerating Data Parallel Training*
(the ring-all-reduce DDP design that displaced parameter servers).
`1910.02054` — *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*
(the three sharding stages).
`2304.11277` — *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel*.
`1909.08053` — *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model
Parallelism* (column/row splitting, one all-reduce per block).
`2205.05198` — *Reducing Activation Recomputation in Large Transformer Models* (sequence
parallelism and selective activation recomputation).
`2104.04473` — *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*
(the PTD-P composition study; the origin of the ordering heuristic in §3.7).
`1811.06965` — *GPipe* (microbatching and the bubble).
`1806.03377` — *PipeDream* (1F1B).
`2401.10241` — *Zero Bubble Pipeline Parallelism*.
`2006.16668` — *GShard* (expert parallelism and the all-to-all).
`2101.03961` — *Switch Transformer* (top-1 routing, capacity factor, token dropping).
`1604.06174` — *Training Deep Nets with Sublinear Memory Cost* (activation checkpointing).
`1710.03740` — *Mixed Precision Training* (the fp32 master-weight ledger of §3.3).
`2410.06511` — *TorchTitan* (the codebase §5 mostly reads).
`2201.12023` — *Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep
Learning* (automatic plan search, 2022).

**Scale, failures, and interconnect.**
`2402.15627` — *MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs*.
`2407.21783` — *The Llama 3 Herd of Models* (failure rates as a first-order design input).
`2411.01137` — *Data movement limits to frontier model training*.

**MoE communication, 2025–2026.**
`2505.11432` — *MegaScale-MoE*.
`2505.13345` — *Occult: Optimizing Collaborative Communication across Experts*.
`2605.11005` — *DisagMoE: Computation-Communication overlapped MoE Training via Disaggregated
AF-Pipe Parallelism* (May 2026).

**Pipeline scheduling, 2025–2026.**
`2503.01328` — *PipeOffload*.
`2504.14519` — *SlimPipe: Memory-Thrifty and Efficient Pipeline Parallelism for Long-Context LLM
Training*.
`2605.24006` — *A Tabular Schedule Abstraction for Communication-Aware Evaluation of
Pipeline-Parallel LLM Training* (May 2026).
`2606.07881` — *Breaking the Bubble: Asynchronous Pipeline Parallel Training with Bounded Weight
Inconsistency* (Jun 2026).

**Plan search and heterogeneity, 2025–2026.**
`2503.17924` — *WLB-LLM: Workload-Balanced 4D Parallelism for Large Language Model Training*.
`2504.00598` — *CFP: Efficient Optimization of Intra-Operator Parallelism Plans*.
`2604.27089` — *AutoSP: Unlocking Long-Context LLM Training Via Compiler-Based Sequence
Parallelism* (Apr 2026).
`2606.16907` — *Tangram: Hiding GPU Heterogeneity for Efficient LLM Parallelization* (Jun 2026).

**Low-communication training.**
`2311.08105` — *DiLoCo: Distributed Low-Communication Training of Language Models*.
`2501.18512` — *Streaming DiLoCo with overlapping communication*.
`2508.15706` — *Communication Efficient LLM Pre-training with SparseLoCo*.
`2606.00271` — *HeLoCo: Efficient asynchronous low-communication training under data and device
heterogeneity* (Jun 2026).

**Context and sequence parallelism (background for §4.1).**
`2310.01889` — *Ring Attention with Blockwise Transformers for Near-Infinite Context*.
`2309.14509` — *DeepSpeed Ulysses*.
`2602.21788` — *Efficient Scaling of LLM Training with Flexible Context Parallelism* (Feb 2026).

**Verification status of this list**, run 2026-07-26 with
`scripts/verify_citations.py curriculum/distributed-training-strategies.md --known
research/reference/papers/anchors.bib`:

```
77 already verified via anchors.bib
36 needed checking  ->  15 resolved, 0 unresolved, 21 unreachable (arXiv HTTP 429)
```

**Zero unresolved.** Nothing in this list failed to resolve; 21 ids are simply unchecked because
arXiv began throttling this repo's verifier earlier in the day (the same condition
`curriculum/README.md` records for the rest of the collection). Unchecked is not the same as
wrong, but the house standard is machine verification, so finish it with:

```
python scripts/verify_citations.py curriculum/distributed-training-strategies.md \
    --known research/reference/papers/anchors.bib \
    --out curriculum/citation-verification.json
```

Note the `--out` argument: without it the script derives an output path by appending to the
*input* path and fails with `FileNotFoundError` when the input is a single file rather than a
directory. That is a real defect in the verifier, found while writing this module, and worth a
failing test before it is fixed.

Ids in the 2025–2026 blocks were surfaced by live search on 2026-07-26 and are cited with the
titles as returned; where a title is quoted it is the title the search returned, not a
paraphrase.
Non-arXiv software cited by name rather than id: DeepEP (GitHub, integrated at
`torchtitan/distributed/deepep/`), torchft (integrated at
`torchtitan/experiments/torchft/`), ROCm issues #6022 and #6034.

### Notes this module must not contradict, and does not

`research/notes/pretraining-recipes.md` §2 (gradient accumulation is our data parallelism;
batch-size advice inverts on one device), §5 (the wall-clock and rent arithmetic this module's
Exercise C plugs into), §9 (the JSONL telemetry schema; a parallel plan adds fields to it).
`research/memory/open-problems-ranked.md` (the envelope; this module upgrades its
"collectives incomplete `[C]`" row to a stronger local `[M]` and proposes the correction rather
than editing the record).
`research/notes/moe-routing-and-failure-modes.md` (routing quality; §3.6 adds the bandwidth
coefficient on `k`).
Track D's sibling modules own the material this one deliberately points at and does not teach:
`checkpointing-and-resumption.md` (§4.4 states only the *format* requirement that a parallel
plan imposes; the DR argument, the RPO arithmetic and the round-trip test belong there),
`determinism-and-reproducibility.md` (§4.5 and §3.4 raise reduction-order non-determinism as a
consequence of the plan; the seeding and bit-exactness discipline belongs there), and
`training-telemetry-as-observability.md` (a parallel plan adds fields to the JSONL schema in
`research/notes/pretraining-recipes.md` §9 — plan degrees, per-collective bytes and wait time,
straggler rank — but the observability design belongs there).
